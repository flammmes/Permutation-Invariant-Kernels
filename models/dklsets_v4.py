import math, os, json, time
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import nn, Tensor
import gpytorch
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DenseLinearOperator
import torch.nn.functional as F  # ADD this at top (two-set file currently lacks it)
from botorch.posteriors.gpytorch import GPyTorchPosterior

# ---------- Utilities ----------

def _to_2d_y(t: Tensor) -> Tensor:
    t_orig = t.shape
    t = t.squeeze()
    if t.dim() == 1: t = t.unsqueeze(-1)
    if t.dim() == 0 and t_orig.numel() > 0: t = t.view(1, 1)
    return t

def _save_state(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)

def _kernel_by_name(name: str, ard_dim: Optional[int]):
    name = name.lower()
    if name in ("rbf", "se", "sqexp", "squared_exponential"):
        return RBFKernel(ard_num_dims=ard_dim if ard_dim and ard_dim > 0 else None)
    if name in ("matern52", "matern_52", "matern"):
        return MaternKernel(nu=2.5, ard_num_dims=ard_dim if ard_dim and ard_dim > 0 else None)
    if name in ("matern32", "matern_32"):
        return MaternKernel(nu=1.5, ard_num_dims=ard_dim if ard_dim and ard_dim > 0 else None)
    raise ValueError(f"Unknown base kernel: {name}")

# ---------- Feature extractors ----------

class MLPFeature(nn.Module):
    """Standard MLP feature extractor for vector inputs."""
    def __init__(self, in_dim: int, latent_dim: int = 64, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., D)
        orig = x.shape[:-1]
        z = self.net(x.reshape(-1, x.size(-1)))       # (prod(orig), latent)
        return z.reshape(*orig, -1)                   # (..., latent)
class MultiPoolDeepSetsFeature(nn.Module):
    """
    Shared-phi Deep Sets with multiple invariant poolings concatenated before rho.
    Poolings supported: 'mean', 'sum', 'max', 'std', 'pmean' (with fixed p_list), 'lse' (log-sum-exp).
    Keeps permutation invariance and small parameter count.
    """
    def __init__(self, n_points: int, point_dim: int, latent_dim: int = 64,
                 phi_hidden: int = 64, rho_hidden: int = 128,
                 pools: List[str] = ("mean", "max", "std", "pmean"),
                 p_list: Optional[List[float]] = (0.0, 2.0),  # 0≈geom, 2≈RMS
                 use_phi_ln: bool = True):
        super().__init__()
        self.n_points = int(n_points)
        self.point_dim = int(point_dim)
        self.pools = list(pools)
        self.p_list = list(p_list) if p_list is not None else []

        # element-wise phi (shared across all pools)
        self.phi = nn.Sequential(
            nn.Linear(point_dim, phi_hidden), nn.GELU(),
            nn.Linear(phi_hidden, phi_hidden), nn.GELU(),
            nn.Linear(phi_hidden, rho_hidden),
        )
        self.phi_ln = nn.LayerNorm(rho_hidden) if use_phi_ln else nn.Identity()

        # compute how wide the concatenation will be
        k = 0
        for name in self.pools:
            if name == "pmean":
                k += len(self.p_list)
            else:
                k += 1
        concat_dim = k * rho_hidden

        # these exist in the one-set version (even if gates/LN are disabled by default)
        self.use_pool_ln = True
        self.pool_ln = nn.ModuleList([nn.LayerNorm(rho_hidden) for _ in range(k)]) if self.use_pool_ln else None

        self.use_pool_gates = False
        self.pool_gate_param = nn.Parameter(torch.zeros(k)) if self.use_pool_gates else None

        # final rho after concatenation
        self.rho = nn.Sequential(
            nn.Linear(concat_dim, rho_hidden), nn.GELU(),
            nn.Linear(rho_hidden, latent_dim),
        )

    @staticmethod
    def _pmean(t: Tensor, p: float, dim: int) -> Tensor:
        # Power-mean on positive features; use softplus to avoid negatives.
        # p=0 -> geometric mean via log-exp trick
        t_pos = F.softplus(t) + 1e-8
        if abs(p) < 1e-6:
            return torch.exp(torch.mean(torch.log(t_pos), dim=dim))
        return torch.pow(torch.mean(torch.pow(t_pos, p), dim=dim), 1.0 / p)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., m*d)
        *lead, D = x.shape
        m, d = self.n_points, self.point_dim
        assert D == m * d, f"MultiPoolDeepSets expects last dim {m*d}, got {D}"
        xs = x.reshape(*lead, m, d)                    # (..., m, d)
        B = int(torch.tensor(lead).prod()) if len(lead) else 1
        xs = xs.reshape(B, m, d)                       # (B, m, d)

        h = self.phi(xs)                               # (B, m, H)
        h = self.phi_ln(h)

        pooled = []
        pool_idx = 0

        def _postprocess(v: Tensor, idx: int) -> Tensor:
            # NOTE: pool_ln exists in one-set but wasn't used there; keep parity.
            if self.pool_gate_param is not None:
                gate = 5.0 * torch.sigmoid(self.pool_gate_param[idx])
                v = v * gate
            return v

        for name in self.pools:
            if name == "mean":
                v = h.mean(dim=1)
                pooled.append(_postprocess(v, pool_idx)); pool_idx += 1

            elif name == "sum":
                v = h.sum(dim=1)
                pooled.append(_postprocess(v, pool_idx)); pool_idx += 1

            elif name == "max":
                v = h.max(dim=1).values
                pooled.append(_postprocess(v, pool_idx)); pool_idx += 1

            elif name == "std":
                v = h.std(dim=1, unbiased=False)
                pooled.append(_postprocess(v, pool_idx)); pool_idx += 1

            elif name == "lse":
                # match one-set: log-mean-exp (scale-insensitive)
                v = torch.logsumexp(h, dim=1) - math.log(h.size(1))
                pooled.append(_postprocess(v, pool_idx)); pool_idx += 1

            elif name == "pmean":
                for p in self.p_list:
                    v = self._pmean(h, p=p, dim=1)
                    pooled.append(_postprocess(v, pool_idx)); pool_idx += 1

            else:
                raise ValueError(f"Unknown pool '{name}'")

        s = torch.cat(pooled, dim=-1)
        z = self.rho(s)
        return z.reshape(*lead, -1)

class EmptySetEmbedding(nn.Module):
    """Returns a (learnable) constant embedding when a set has zero points."""
    def __init__(self, latent_dim: int, learnable: bool = True):
        super().__init__()
        if learnable:
            self.embed = nn.Parameter(torch.zeros(latent_dim))
        else:
            self.register_buffer("embed", torch.zeros(latent_dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., 0) typically; we only use the leading batch shape
        lead = x.shape[:-1]
        e = self.embed.to(dtype=x.dtype, device=x.device)
        # reshape to broadcast over lead
        view_shape = (1,) * len(lead) + (e.numel(),)
        return e.view(*view_shape).expand(*lead, -1)

class TwoSetTwoVecDeepSetsFeature(nn.Module):
    """
    Feature extractor for inputs consisting of:
      - a vector part (order-sensitive) of dimension vec_dim
      - a first set (e.g., injector coordinates) with n_inj points of dimension point_dim
      - a second set (e.g., producer coordinates) with n_prod points of dimension point_dim

    The two sets are each processed by their own Deep Sets encoder (permutation-invariant
    *within* each set). The vector part is processed by an MLP. The three embeddings
    are concatenated and fused to a final latent embedding of size latent_dim.

    Expected flattened input layout:
      x = [ vec | inj_set_flat | prod_set_flat ]
      where inj_set_flat has length n_inj * point_dim and prod_set_flat has length n_prod * point_dim.
    """
    def __init__(
        self,
        vec_dim: int,
        n_inj: int,
        n_prod: int,
        point_dim: int = 2,
        latent_dim: int = 64,
        # vector MLP config
        mlp_hidden: int = 64,
        # deep-sets config
        phi_hidden: int = 64,
        rho_hidden: int = 128,
        pools: List[str] = ("mean", "max", "std", "pmean"),
        p_list: Optional[List[float]] = (0.0, 2.0),
        # fusion
        fuse_hidden: int = 128,
        use_phi_ln: bool = True,
    ):
        super().__init__()
        self.vec_dim = int(vec_dim)
        self.n_inj = int(n_inj)
        self.n_prod = int(n_prod)
        self.point_dim = int(point_dim)

        self.vec_feat = MLPFeature(in_dim=self.vec_dim, latent_dim=latent_dim, hidden=mlp_hidden)

        if self.n_inj > 0:
            self.inj_feat = MultiPoolDeepSetsFeature(
                n_points=self.n_inj,
                point_dim=self.point_dim,
                latent_dim=latent_dim,
                phi_hidden=phi_hidden,
                rho_hidden=rho_hidden,
                pools=list(pools),
                p_list=list(p_list) if p_list is not None else [],
                use_phi_ln=use_phi_ln,
            )
        else:
            self.inj_feat = EmptySetEmbedding(latent_dim=latent_dim, learnable=True)

        if self.n_prod > 0:
            self.prod_feat = MultiPoolDeepSetsFeature(
                n_points=self.n_prod,
                point_dim=self.point_dim,
                latent_dim=latent_dim,
                phi_hidden=phi_hidden,
                rho_hidden=rho_hidden,
                pools=list(pools),
                p_list=list(p_list) if p_list is not None else [],
                use_phi_ln=use_phi_ln,
            )
        else:
            self.prod_feat = EmptySetEmbedding(latent_dim=latent_dim, learnable=True)

        # Relational (injector-producer) DeepSets encoder
        # R has n_inj*n_prod points, each in R^2 (displacement vectors)
        if self.n_inj > 0 and self.n_prod > 0:
            self.rel_feat = MultiPoolDeepSetsFeature(
                n_points=self.n_inj * self.n_prod,
                point_dim=self.point_dim,          # displacement lives in same dim as points (2)
                latent_dim=latent_dim,
                phi_hidden=phi_hidden,
                rho_hidden=rho_hidden,
                pools=list(pools),
                p_list=list(p_list) if p_list is not None else [],
                use_phi_ln=use_phi_ln,
            )
        else:
            self.rel_feat = EmptySetEmbedding(latent_dim=latent_dim, learnable=True)


            
        self.fuse = nn.Sequential(
            nn.Linear(4 * latent_dim, fuse_hidden), nn.GELU(),
            nn.Linear(fuse_hidden, latent_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        *lead, D = x.shape
        expected = self.vec_dim + (self.n_inj + self.n_prod) * self.point_dim
        if D != expected:
            raise RuntimeError(
                f"TwoSetTwoVecDeepSetsFeature expects last dim {expected} "
                f"(vec_dim={self.vec_dim} + (n_inj+n_prod)*point_dim), got {D}."
            )

        idx = 0
        x_vec = x[..., idx: idx + self.vec_dim]
        idx += self.vec_dim

        inj_len = self.n_inj * self.point_dim
        x_inj = x[..., idx: idx + inj_len]
        idx += inj_len

        prod_len = self.n_prod * self.point_dim
        x_prod = x[..., idx: idx + prod_len]
        # reshape flat sets to point tensors
        x_inj_pts = x_inj.reshape(*lead, self.n_inj, self.point_dim)   # (..., n_inj, 2)
        x_prd_pts = x_prod.reshape(*lead, self.n_prod, self.point_dim) # (..., n_prod, 2)

        if self.n_inj > 0 and self.n_prod > 0:
            # (..., n_inj, n_prod, 2)
            rel = x_prd_pts.unsqueeze(-3) - x_inj_pts.unsqueeze(-2)
            # (..., n_inj*n_prod*2)  (since DeepSetsFeature expects flat m*d)
            x_rel = rel.reshape(*lead, self.n_inj * self.n_prod * self.point_dim)
        else:
            x_rel = x[..., :0]  # empty placeholder


        z_vec = self.vec_feat(x_vec)
        z_inj = self.inj_feat(x_inj)
        z_prd = self.prod_feat(x_prod)
        z_rel = self.rel_feat(x_rel)

        z = torch.cat([z_vec, z_inj, z_prd, z_rel], dim=-1)
        return self.fuse(z)



class _DKLExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor,
                 feature_extractor: nn.Module, base_kernel: gpytorch.kernels.Kernel,
                 likelihood: GaussianLikelihood):
        super().__init__(train_x, train_y, likelihood)
        self.feat = feature_extractor
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(base_kernel)

    def forward(self, x: Tensor) -> MultivariateNormal:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, D)
        z = self.feat(x)       # works for (N,D) and (B,q,D)

        mean = self.mean_module(z)                     # (..., n)
        cov  = self.covar_module(z)                    # LinearOperator over (..., n, n)
        return MultivariateNormal(mean, cov)

# ---------- Common training wrapper ----------

class _DKLWrapper(torch.nn.Module):
    """
    Light BoTorch-compatible wrapper exposing .posterior and .fit_and_save
    """
    def __init__(self, model_args: dict, input_dim: int, output_dim: int, device: torch.device,
                 feature_extractor: nn.Module):
        super().__init__()
        if output_dim != 1:
            raise RuntimeError("DKL models here are single-output only.")
        self.output_dim = output_dim
        self.input_dim  = input_dim
        self.device = device

        # hyperparams (defaults sensible for BO loops)
        self.latent_dim     = int(model_args.get("latent_dim", 64))
        self.lr             = float(model_args.get("lr", 3e-3))
        self.weight_decay   = float(model_args.get("weight_decay", 1e-4))
        self.train_epochs   = int(model_args.get("train_epochs", 250))
        self.pretrain_epochs= int(model_args.get("pretrain_epochs", 0))  # small warmup is optional
        self.clip_grad      = float(model_args.get("clip_grad", 1.0))
        self.learn_noise    = bool(model_args.get("learn_noise", True))
        self.kernel_name    = model_args.get("kernel", "matern52")
        self.ard_in_latent  = bool(model_args.get("ard_in_latent", True))
        self.jitter_schedule= list(model_args.get("jitter_schedule", [1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]))
        self.seed           = int(model_args.get("seed", 1234))

        torch.manual_seed(self.seed)

        # instantiate likelihood + kernel in latent space
        ard_dim = self.latent_dim if self.ard_in_latent else None
        base_k = _kernel_by_name(self.kernel_name, ard_dim=ard_dim)

        self.likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-4))
        if not self.learn_noise:

            self.likelihood.noise_covar.raw_noise.requires_grad_(False)

        self.feature = feature_extractor.to(device)


        # gp model built at fit-time once data are known
        self.gp: Optional[_DKLExactGP] = None

        # logs
        self.train_log = {}
        self.diag = {}

    # ----- BoTorch-ish API -----
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable] = None,
        **kwargs: Any,
    ):
        if self.gp is None:
            raise RuntimeError("Model not yet fit.")
        self.gp.eval()
        self.likelihood.eval()

        with gpytorch.settings.fast_pred_var():
            post_latent = self.gp(X.to(self.device))  # gpytorch MVN in standardized space

            # map back: y = mu + sigma * z
            mean = self.y_mean + self.y_std * post_latent.mean
            cov  = (self.y_std ** 2) * post_latent.covariance_matrix

            mvn = MultivariateNormal(mean, DenseLinearOperator(cov))

            if observation_noise:
                mvn = self.likelihood(mvn)

            post = GPyTorchPosterior(mvn)  # <-- key: adds explicit output dim m=1

            if posterior_transform is not None:
                post = posterior_transform(post)

            return post

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([])

    @property
    def num_outputs(self) -> int:
        return 1

    def _feature_output_dim(self) -> int:
        """Best-effort way to discover the latent dim produced by the feature extractor."""
        # Case 1: feature has a .net (your MLPFeature)
        if hasattr(self.feature, "net") and isinstance(self.feature.net, nn.Sequential):
            last = self.feature.net[-1]
            if isinstance(last, nn.Linear):
                return int(last.out_features)

        # Case 2: feature has a .rho (your DeepSetsFeature)
        if hasattr(self.feature, "rho") and isinstance(self.feature.rho, nn.Sequential):
            last = self.feature.rho[-1]
            if isinstance(last, nn.Linear):
                return int(last.out_features)

        # Fallback: run a tiny forward on zeros
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_dim, device=self.device)
            z = self.feature(dummy)
            return int(z.shape[-1])

    def _pretrain_feature(self, X: Tensor, Y: Tensor, epochs: int) -> None:
        if epochs <= 0:
            return
        in_dim = self._feature_output_dim()
        head = nn.Linear(in_dim, 1).to(self.device)

        opt = torch.optim.Adam(
            list(self.feature.parameters()) + list(head.parameters()),
            lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        self.feature.train(); head.train()
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            # You can just remove autocast entirely since enabled=False was a no-op
            with torch.amp.autocast(device_type="cuda", enabled=False):
                z = self.feature(X)
                yhat = head(z).squeeze(-1)
                loss = loss_fn(yhat, Y.squeeze(-1))
            loss.backward()
            if self.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.feature.parameters()) + list(head.parameters()),
                    self.clip_grad
                )
            opt.step()
        del head

    def fit_and_save(self, train_x: Tensor, train_y: Tensor, save_dir: Optional[str]):
        X = train_x.to(self.device)
        Y = _to_2d_y(train_y).to(self.device)
        Y1D = Y.squeeze(-1)
        self.y_mean = Y1D.mean()
        raw_std = Y1D.std()
        self.y_std = raw_std.clamp_min(1e-4 * (Y1D.abs().mean() + 1e-8))

        Y1D = ((Y - self.y_mean) / self.y_std).squeeze(-1)

        # (re)build model with current training tensors
        ard_dim = self.latent_dim if self.ard_in_latent else None
        base_k = _kernel_by_name(self.kernel_name, ard_dim)
        self.gp = _DKLExactGP(X, Y1D, self.feature, base_k.to(self.device), self.likelihood.to(self.device)).to(self.device)

        # optional warm start of the feature extractor
        self._pretrain_feature(X, Y, self.pretrain_epochs)

        opt = torch.optim.Adam(self.gp.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.gp).to(self.device)

        t0 = time.time()
        last_loss = None
        for jitter in self.jitter_schedule:
            try:
                for ep in range(self.train_epochs):
                    self.gp.train(); self.likelihood.train()
                    opt.zero_grad(set_to_none=True)
                    with gpytorch.settings.cholesky_jitter(jitter):
                        output = self.gp(X)
                        loss = -mll(output, Y1D)
                    loss.backward()
                    if self.clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(self.gp.parameters(), self.clip_grad)
                    opt.step()
                    last_loss = float(loss.detach().cpu())
                    if (ep + 1) % 400 == 0 or ep == 0 or (ep + 1) == self.train_epochs:
                        with torch.no_grad():
                            base = self.gp.covar_module.base_kernel
                            if isinstance(base, (RBFKernel, MaternKernel)):
                                ls = base.lengthscale.detach().cpu().reshape(-1)
                                ls_med = float(ls.median().item())
                            else:
                                ls_med = float('nan')
                            noise_val = float(self.likelihood.noise.detach().cpu())
                            curr_mll = float(-loss.detach().cpu())  # MLL (positive)
                        print(f"[DKL] ep {ep+1:4d} | jitter={jitter:.1e} | "
                              f"MLL={curr_mll:.4f} | noise={noise_val:.3e} | ls_med={ls_med:.4f}")



                # if we got here, this jitter level converged — break
                break
            except RuntimeError as e:
                # try next jitter
                continue
        fit_time = time.time() - t0

        # log some hypers
        with torch.no_grad():
            self.gp.eval(); self.likelihood.eval()
            base = self.gp.covar_module.base_kernel
            if isinstance(base, (RBFKernel, MaternKernel)):
                ls = base.lengthscale.detach().cpu().reshape(-1).tolist()
            else:
                ls = None
            self.train_log = {
                "fit_time_sec": fit_time,
                "final_mll": -last_loss if last_loss is not None else None,
                "noise": float(self.likelihood.noise.detach().cpu()),
                "outputscale": float(self.gp.covar_module.outputscale.detach().cpu()),
                "lengthscale": ls,
            }

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.gp.state_dict(), os.path.join(save_dir, "model.pt"))
            with open(os.path.join(save_dir, "dkl_diagnostics.json"), "w") as f:
                json.dump({"train_log": self.train_log}, f, indent=2)

# ---------- Public classes wired to your main() ----------

class SingleTaskDKL:
    """
    DKL with a standard MLP feature extractor.
    Signature matches initialize_model(..) usage in your main.
    """
    def __init__(self, model_args: dict, input_dim: int, output_dim: int, device: torch.device):
        latent_dim = int(model_args.get("latent_dim", 64))
        hidden = int(model_args.get("mlp_hidden", 64))
        feat = MLPFeature(in_dim=input_dim, latent_dim=latent_dim, hidden=hidden)
        self._impl = _DKLWrapper(model_args, input_dim, output_dim, device, feat)

    # delegate BoTorch-ish API
    def posterior(self, *a, **kw): return self._impl.posterior(*a, **kw)
    @property
    def batch_shape(self): return self._impl.batch_shape
    @property
    def num_outputs(self): return self._impl.num_outputs
    def fit_and_save(self, *a, **kw): return self._impl.fit_and_save(*a, **kw)
    def eval(self): self._impl.eval()



class SingleTaskDKLDeepSets:
    def __init__(self, model_args, input_dim, output_dim, device):
        # If (n_inj, n_prod) are provided we assume a 2-vector + 2-set input layout:
        #   x = [ vec | inj_set_flat | prod_set_flat ]
        # Otherwise we fall back to the original single-set layout (n_points, point_dim).
        latent_dim = int(model_args.get("latent_dim", 64))
        phi_hidden = int(model_args.get("phi_hidden", 64))
        rho_hidden = int(model_args.get("rho_hidden", 128))

        pools = model_args.get("pools", ["mean", "max", "std", "pmean"])
        p_list = model_args.get("p_list", [0.0, 2.0])   # geometric & RMS

        if ("n_inj" in model_args) or ("n_prod" in model_args):
            n_inj = int(model_args.get("n_inj", 0))
            n_prod = int(model_args.get("n_prod", 0))
            if n_inj < 0 or n_prod < 0:
                raise ValueError("n_inj and n_prod must be >= 0.")

            vec_dim = int(model_args.get("vec_dim", 2))
            point_dim = int(model_args.get("point_dim", 2))
            expected = vec_dim + (n_inj + n_prod) * point_dim
            assert expected == input_dim, f"Expected input_dim={expected} (vec_dim + (n_inj+n_prod)*point_dim), got {input_dim}"

            mlp_hidden = int(model_args.get("mlp_hidden", 64))
            fuse_hidden = int(model_args.get("fuse_hidden", 128))

            feat = TwoSetTwoVecDeepSetsFeature(
                vec_dim=vec_dim,
                n_inj=n_inj,
                n_prod=n_prod,
                point_dim=point_dim,
                latent_dim=latent_dim,
                mlp_hidden=mlp_hidden,
                phi_hidden=phi_hidden,
                rho_hidden=rho_hidden,
                pools=pools,
                p_list=p_list,
                fuse_hidden=fuse_hidden,
                use_phi_ln=True,
            )
        else:
            m = int(model_args.get("n_points", 5))
            d_elem = int(model_args.get("point_dim", input_dim // max(m, 1)))
            assert m * d_elem == input_dim, f"Single-set DeepSets expects input_dim=m*point_dim, got {input_dim}"

            feat = MultiPoolDeepSetsFeature(
                n_points=m, point_dim=d_elem, latent_dim=latent_dim,
                phi_hidden=phi_hidden, rho_hidden=rho_hidden,
                pools=pools, p_list=p_list, use_phi_ln=True
            )

        self._impl = _DKLWrapper(model_args, input_dim, output_dim, device, feat)

    # delegate
    def posterior(self, *a, **kw): return self._impl.posterior(*a, **kw)
    @property
    def batch_shape(self): return self._impl.batch_shape
    @property
    def num_outputs(self): return self._impl.num_outputs
    def fit_and_save(self, *a, **kw): return self._impl.fit_and_save(*a, **kw)
    def eval(self): self._impl.eval()
