# models/gp_set_kernels_de_v2.py
"""
Permutation-invariant engineered kernels for set inputs:
- DoubleSumSetKernel (DS)
- DeepEmbeddingSetKernel (DE)

This v2 adds support for "2 vectors + 2 sets" inputs used in the paper-style BO experiments:
  x = [v (vec_dim) | set_inj (n_inj * point_dim) | set_prod (n_prod * point_dim)]

We combine component kernels via either product (default) or additive composition:
  combine = "product"  -> k = k_vec * k_inj * k_prod
  combine = "additive" -> k = k_vec + k_inj + k_prod

Component selection is done via gpytorch's `active_dims`, so the original single-set behavior
is preserved when model_args do not specify n_inj/n_prod.
"""
from linear_operator.utils.errors import NotPSDError

from typing import Optional, List, Any, Dict, Tuple
import math
import torch
from torch import Tensor
import botorch
from botorch.posteriors import Posterior
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import (
    Kernel,
    ScaleKernel,
    MaternKernel,
    RBFKernel,
    ProductKernel,
    AdditiveKernel,
)
from gpytorch.priors import GammaPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
import gpytorch

import json
import random
import time


# ---- Helpers ----

def _to_2d(t: Tensor) -> Tensor:
    t_orig_shape = t.shape
    t = t.squeeze()
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    if t.dim() == 0 and t_orig_shape.numel() > 0:
        t = t.view(1, 1)
    elif t.dim() != 2 or (t.size(-1) != 1 and t.size(0) != 0):
        if t.numel() == 0 and t.size(-1) != 1:
            t = t.view(0, 1)
        elif t.numel() > 0:
            raise RuntimeError(f"Y shape after squeeze from {t_orig_shape} is {t.shape}, expected (n,1)")
    return t


def _pairwise_rbf(xa: Tensor, xb: Tensor, ell_x: Tensor) -> Tensor:
    # xa: (..., m, d), xb: (..., m, d) or (..., m2, d)
    # returns: (..., m, m2) with exp(-||xi - xj'||^2 / (2 ell_x^2))
    diff = xa.unsqueeze(-2) - xb.unsqueeze(-3)  # (..., m, m2, d)
    dist2 = (diff ** 2).sum(dim=-1)
    ell = torch.clamp(ell_x, min=1e-8)
    return torch.exp(-0.5 * dist2 / (ell * ell))


def _k0_self_mean(X: Tensor, m: int, d_elem: int, ell_x: Tensor) -> Tensor:
    # X: (N, D) where D = m * d_elem
    N, D = X.shape
    sets = X.reshape(N, m, d_elem)
    K = _pairwise_rbf(sets, sets, ell_x)            # (N, m, m)
    return K.mean(dim=(-2, -1))                     # (N,)


@torch.no_grad()
def _spectral_snapshot(covar_module: Kernel, X: Tensor, max_n: int = 200) -> dict:
    n = min(X.size(0), max_n)
    idx = torch.randperm(X.size(0), device=X.device)[:n]
    Xs = X.index_select(0, idx)
    K = covar_module(Xs).evaluate()
    if hasattr(K, "to_dense"):
        K = K.to_dense()
    evals = torch.linalg.eigvalsh(K)
    evals = torch.clamp(evals, min=0.0)
    ev_np = evals.detach().cpu()
    return {
        "n": int(n),
        "eig_min": float(ev_np.min().item()),
        "eig_median": float(ev_np.median().item()),
        "eig_max": float(ev_np.max().item()),
        "cond_est": float((ev_np.max() / ev_np.clamp_min(1e-12).min()).item()),
    }


@torch.no_grad()
def _perm_invariance_check_two_sets(
    covar_module: Kernel,
    X: Tensor,
    vec_dim: int,
    n_inj: int,
    n_prod: int,
    point_dim: int,
    trials: int = 5,
    permute_inj: bool = True,
    permute_prod: bool = True,
) -> float:
    """
    Returns max |k(x_i, x_j) - k(x_i_perm, x_j)| where x_i_perm permutes elements within
    injector set and/or producer set blocks (vector block unchanged).
    """
    if X.size(0) < 2:
        return float("nan")

    N, D = X.shape
    inj_start = vec_dim
    inj_end = inj_start + n_inj * point_dim
    prod_start = inj_end
    prod_end = prod_start + n_prod * point_dim

    if prod_end > D:
        raise RuntimeError(
            f"Input dim={D} too small for vec_dim={vec_dim}, n_inj={n_inj}, n_prod={n_prod}, point_dim={point_dim}. "
            f"Need >= {prod_end}."
        )

    max_abs_diff = 0.0
    for _ in range(trials):
        i, j = random.randrange(N), random.randrange(N)
        xi = X[i].clone()
        xj = X[j].clone()

        # permute injector set (block: inj_start:inj_end)
        if permute_inj and n_inj > 1:
            Si = xi[inj_start:inj_end].reshape(n_inj, point_dim)
            perm = torch.randperm(n_inj, device=X.device)
            Si_p = Si[perm].reshape(-1)
            xi[inj_start:inj_end] = Si_p

        # permute producer set (block: prod_start:prod_end)
        if permute_prod and n_prod > 1:
            Sp = xi[prod_start:prod_end].reshape(n_prod, point_dim)
            perm = torch.randperm(n_prod, device=X.device)
            Sp_p = Sp[perm].reshape(-1)
            xi[prod_start:prod_end] = Sp_p

        # shape to (1,1,D)
        xi_b = xi.reshape(1, 1, -1)
        xj_b = xj.reshape(1, 1, -1)
        xip_b = X[i].reshape(1, 1, -1)  # original
        # k(original_i, j) vs k(permuted_i, j)
        K_ij = covar_module(xip_b, xj_b).evaluate()
        K_p  = covar_module(xi_b,  xj_b).evaluate()

        if hasattr(K_ij, "to_dense"):
            K_ij = K_ij.to_dense()
            K_p  = K_p.to_dense()

        kij = K_ij.squeeze()
        kp  = K_p.squeeze()
        max_abs_diff = max(max_abs_diff, float((kij - kp).abs().item()))

    return max_abs_diff


def _vec_kernel_by_name(name: str, ard_dim: int, active_dims: torch.Tensor) -> Kernel:
    name = str(name).lower()
    if name in ("rbf", "se", "sqexp", "squared_exponential"):
        return RBFKernel(ard_num_dims=ard_dim, active_dims=active_dims)
    if name in ("matern52", "matern_52", "matern"):
        return MaternKernel(nu=2.5, ard_num_dims=ard_dim, active_dims=active_dims)
    if name in ("matern32", "matern_32"):
        return MaternKernel(nu=1.5, ard_num_dims=ard_dim, active_dims=active_dims)
    raise ValueError(f"Unknown vector kernel: {name}")


def _build_two_set_two_vec_kernel_ds_or_de(
    *,
    kind: str,
    vec_dim: int,
    n_inj: int,
    n_prod: int,
    point_dim: int,
    vec_kernel_name: str,
    combine: str,
):
    kind = kind.lower()
    combine = str(combine).lower()
    if combine not in ("product", "additive"):
        raise ValueError(f"combine must be 'product' or 'additive', got {combine}")

    vec_active  = torch.arange(0, vec_dim, dtype=torch.long)
    inj_active  = torch.arange(vec_dim, vec_dim + n_inj * point_dim, dtype=torch.long)
    prod_active = torch.arange(vec_dim + n_inj * point_dim,
                               vec_dim + (n_inj + n_prod) * point_dim,
                               dtype=torch.long)

    vec_k = _vec_kernel_by_name(vec_kernel_name, ard_dim=vec_dim, active_dims=vec_active)

    kernels = [vec_k]
    comps = {"vec": vec_k, "inj": None, "prod": None}

    if n_inj > 0:
        if kind == "ds":
            inj_k = DoubleSumSetKernel(n_points=n_inj, active_dims=inj_active)
        elif kind == "de":
            inj_k = DeepEmbeddingSetKernel(n_points=n_inj, active_dims=inj_active)
        else:
            raise ValueError(f"Unknown kind: {kind}")
        kernels.append(inj_k)
        comps["inj"] = inj_k

    if n_prod > 0:
        if kind == "ds":
            prod_k = DoubleSumSetKernel(n_points=n_prod, active_dims=prod_active)
        elif kind == "de":
            prod_k = DeepEmbeddingSetKernel(n_points=n_prod, active_dims=prod_active)
        else:
            raise ValueError(f"Unknown kind: {kind}")
        kernels.append(prod_k)
        comps["prod"] = prod_k

    # Combine safely even if we only have 1 component
    if len(kernels) == 1:
        base = kernels[0]
    else:
        base = ProductKernel(*kernels) if combine == "product" else AdditiveKernel(*kernels)

    return base, comps


# ---- Kernels ----

class DoubleSumSetKernel(Kernel):
    """
    DS kernel: k0(S, S') = 1/(|S||S'|) sum_{x in S} sum_{x' in S'} k_X(x, x')
    using a Gaussian base kernel k_X with lengthscale ell_x.
    """
    is_stationary = False
    def __init__(self, n_points: int, **kwargs):
        super().__init__(**kwargs)
        self.n_points = int(n_points)
        self.register_parameter("raw_ell_x", torch.nn.Parameter(torch.log(torch.tensor(0.5, dtype=torch.float64))))
        self.register_constraint("raw_ell_x", GreaterThan(-10.0))

    @property
    def ell_x(self) -> Tensor:
        return torch.exp(self.raw_ell_x)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor,
        diag: bool = False, last_dim_is_batch: bool = False, **params
    ) -> torch.Tensor:
        if diag:
            orig_dim = x1.dim()
            x1b = x1.unsqueeze(0) if orig_dim == 2 else x1
            B, N, D = x1b.shape
            m = int(self.n_points)
            if D % m != 0:
                raise RuntimeError(f"Expected last dim to be a multiple of m={m}, got D={D}")
            d_elem = D // m
            x_sets = x1b.reshape(B * N, m, d_elem)
            k_ss = _pairwise_rbf(x_sets, x_sets, self.ell_x).mean(dim=(-2, -1))
            out = k_ss.view(B, N)
            return out.squeeze(0) if orig_dim == 2 else out

        orig_x1_dim, orig_x2_dim = x1.dim(), x2.dim()
        if orig_x1_dim == 2:
            x1 = x1.unsqueeze(0)
        if orig_x2_dim == 2:
            x2 = x2.unsqueeze(0)

        if x1.size(0) != x2.size(0):
            if x1.size(0) == 1:
                x1 = x1.expand(x2.size(0), -1, -1)
            elif x2.size(0) == 1:
                x2 = x2.expand(x1.size(0), -1, -1)
            else:
                raise RuntimeError(f"Incompatible batch sizes: {x1.shape} vs {x2.shape}")

        B, N1, D = x1.shape
        N2 = x2.size(1)
        m = int(self.n_points)
        if D % m != 0:
            raise RuntimeError(f"Expected last dim to be a multiple of m={m}, got D={D}")
        d_elem = D // m

        x1_sets = x1.reshape(B, N1, m, d_elem)
        x2_sets = x2.reshape(B, N2, m, d_elem)

        xa = x1_sets.unsqueeze(2).expand(B, N1, N2, m, d_elem)
        xb = x2_sets.unsqueeze(1).expand(B, N1, N2, m, d_elem)
        K_pair_mean = _pairwise_rbf(
            xa.reshape(-1, m, d_elem), xb.reshape(-1, m, d_elem), self.ell_x
        ).mean(dim=(-2, -1)).reshape(B, N1, N2)

        if orig_x1_dim == 2 and orig_x2_dim == 2 and K_pair_mean.size(0) == 1:
            K_pair_mean = K_pair_mean.squeeze(0)
        return K_pair_mean


class DeepEmbeddingSetKernel(Kernel):
    """
    DE kernel: k_DE(S,S') = exp(-0.5 d_E^2(S,S') / ell_h^2),
    d_E^2 = k0(S,S) + k0(S',S') - 2 k0(S,S'), with k0 the DS kernel base (Gaussian w/ ell_x).
    Learnable params: ell_x (base), ell_h (radial).
    """
    is_stationary = False
    def __init__(self, n_points: int, **kwargs):
        super().__init__(**kwargs)
        self.n_points = int(n_points)
        self.register_parameter("raw_ell_x", torch.nn.Parameter(torch.log(torch.tensor(0.5, dtype=torch.float64))))
        self.register_parameter("raw_ell_h", torch.nn.Parameter(torch.log(torch.tensor(0.5, dtype=torch.float64))))
        self.register_constraint("raw_ell_x", GreaterThan(-10.0))
        self.register_constraint("raw_ell_h", GreaterThan(-10.0))

    @property
    def ell_x(self) -> Tensor:
        return torch.exp(self.raw_ell_x)

    @property
    def ell_h(self) -> Tensor:
        return torch.exp(self.raw_ell_h)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor,
        diag: bool = False, last_dim_is_batch: bool = False, **params
    ) -> torch.Tensor:
        if diag:
            return x1.new_ones(*x1.shape[:-1])

        orig_x1_dim, orig_x2_dim = x1.dim(), x2.dim()
        if orig_x1_dim == 2:
            x1 = x1.unsqueeze(0)
        if orig_x2_dim == 2:
            x2 = x2.unsqueeze(0)

        if x1.size(0) != x2.size(0):
            if x1.size(0) == 1:
                x1 = x1.expand(x2.size(0), -1, -1)
            elif x2.size(0) == 1:
                x2 = x2.expand(x1.size(0), -1, -1)
            else:
                raise RuntimeError(f"Incompatible batch sizes: {x1.shape} vs {x2.shape}")

        B, N1, D = x1.shape
        N2 = x2.size(1)
        m = int(self.n_points)
        if D % m != 0:
            raise RuntimeError(f"Expected last dim to be a multiple of m={m}, got D={D}")
        d_elem = D // m

        x1_sets = x1.reshape(B, N1, m, d_elem)
        x2_sets = x2.reshape(B, N2, m, d_elem)

        xa_self = x1_sets.reshape(B * N1, m, d_elem)
        xb_self = x2_sets.reshape(B * N2, m, d_elem)
        K_xx = _pairwise_rbf(xa_self, xa_self, self.ell_x).mean(dim=(-2, -1)).reshape(B, N1)
        K_yy = _pairwise_rbf(xb_self, xb_self, self.ell_x).mean(dim=(-2, -1)).reshape(B, N2)

        xa = x1_sets.unsqueeze(2).expand(B, N1, N2, m, d_elem)
        xb = x2_sets.unsqueeze(1).expand(B, N1, N2, m, d_elem)
        K_xy = _pairwise_rbf(
            xa.reshape(-1, m, d_elem), xb.reshape(-1, m, d_elem), self.ell_x
        ).mean(dim=(-2, -1)).reshape(B, N1, N2)

        d2 = (K_xx.unsqueeze(2) + K_yy.unsqueeze(1) - 2.0 * K_xy).clamp_min(1e-12)
        ell_h = torch.clamp(self.ell_h, min=1e-8)
        K = torch.exp(-0.5 * d2 / (ell_h * ell_h))

        if orig_x1_dim == 2 and orig_x2_dim == 2 and K.size(0) == 1:
            K = K.squeeze(0)
        return K


# ---- Model wrappers ----

class _BaseSingleTaskSetGP(botorch.models.model.Model):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.gp = None

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Any] = None,
        **kwargs: Any
    ) -> Posterior:
        if self.gp is None:
            raise RuntimeError("GP not fit yet.")
        return self.gp.posterior(X, output_indices, observation_noise, posterior_transform, **kwargs)

    @property
    def batch_shape(self) -> torch.Size:
        return self.gp.batch_shape if self.gp is not None else torch.Size()

    @property
    def num_outputs(self) -> int:
        return self.gp.num_outputs if self.gp is not None else self.output_dim


def _detect_two_set_layout(model_args: dict) -> Optional[dict]:
    if ("n_inj" in model_args) and ("n_prod" in model_args):
        return {
            "vec_dim": int(model_args.get("vec_dim", 2)),
            "n_inj": int(model_args["n_inj"]),
            "n_prod": int(model_args["n_prod"]),
            "point_dim": int(model_args.get("point_dim", 2)),
            "vec_kernel": str(model_args.get("vec_kernel", "matern52")),
            "combine": str(model_args.get("combine", "product")),
        }
    return None


class SingleTaskGP_DS(_BaseSingleTaskSetGP):
    """
    DS-only set kernel model.
    - If model_args has n_inj/n_prod, uses 2 vectors + 2 sets composite kernel.
    - Else, falls back to the original single-set DS kernel with n_points.
    """
    def __init__(self, model_args: dict, input_dim: int, output_dim: int):
        super().__init__(output_dim)
        self.model_args = dict(model_args)
        self.input_dim = int(input_dim)

        layout = _detect_two_set_layout(model_args)
        self.layout = layout

        if layout is None:
            self.n_points = int(model_args.get("n_points", 5))
        else:
            self.n_points = None  # not used in multi-set mode

        self.train_log = {
            "fit_time_sec": None,
            "mll": None,
            "noise": None,
            "outputscale": None,
            "ell_x": None,   # scalar (single-set) or dict (multi-set)
        }
        self.diag = {
            "E_k0_self_mean": None,              # scalar or dict
            "perm_invariance_max_abs_diff": None,
            "spectral_snapshot": None,
        }

    def fit_and_save(self, train_x: Tensor, train_y: Tensor, save_dir: Optional[str]):
        if self.output_dim > 1:
            raise RuntimeError("SingleTaskGP_DS is single-output only.")
        current_train_x = train_x.squeeze(0) if train_x.dim() == 3 and train_x.shape[0] == 1 else train_x
        if current_train_x.dim() != 2:
            raise RuntimeError(f"train_x shape after squeeze is {current_train_x.shape}, expected (n,d)")
        if current_train_x.size(-1) != self.input_dim:
            raise RuntimeError(f"input_dim mismatch: expected {self.input_dim}, got {current_train_x.size(-1)}")
        current_train_y = _to_2d(train_y)

        # Build kernel
        components = None
        if self.layout is None:
            base = DoubleSumSetKernel(n_points=self.n_points).to(current_train_x)
        else:
            vec_dim = self.layout["vec_dim"]
            n_inj = self.layout["n_inj"]
            n_prod = self.layout["n_prod"]
            point_dim = self.layout["point_dim"]

            expected = vec_dim + (n_inj + n_prod) * point_dim
            if current_train_x.size(-1) != expected:
                raise RuntimeError(f"Expected input_dim={expected} for vec_dim={vec_dim}, n_inj={n_inj}, n_prod={n_prod}, point_dim={point_dim}, got {current_train_x.size(-1)}")

            base, components = _build_two_set_two_vec_kernel_ds_or_de(
                kind="ds",
                vec_dim=vec_dim,
                n_inj=n_inj,
                n_prod=n_prod,
                point_dim=point_dim,
                vec_kernel_name=self.layout["vec_kernel"],
                combine=self.layout["combine"],
            )
            base = base.to(current_train_x)

        covar = ScaleKernel(base, outputscale_prior=GammaPrior(2.0, 0.15))
        self.gp = botorch.models.SingleTaskGP(
            train_X=current_train_x, train_Y=current_train_y,
            covar_module=covar, outcome_transform=Standardize(m=1)
        ).to(current_train_x)

        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp).to(current_train_x)

        t0 = time.time()
        try:
            with gpytorch.settings.cholesky_jitter(1e-4):
                botorch.fit.fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"  [Warning] L-BFGS fit failed: {e}. Falling back to Adam.")
            optimizer = torch.optim.Adam(self.gp.parameters(), lr=0.1)
            self.gp.train(); mll.train()
            for _ in range(100):
                optimizer.zero_grad()
                output = self.gp(current_train_x)
                loss = -mll(output, current_train_y.squeeze())
                if torch.isnan(loss): break
                loss.backward()
                optimizer.step()
        fit_time = time.time() - t0

        with torch.no_grad():
            noise = float(self.gp.likelihood.noise.detach().cpu())
            outscale = float(self.gp.covar_module.outputscale.detach().cpu())

            with torch.no_grad():
                noise = float(self.gp.likelihood.noise.detach().cpu())
                outscale = float(self.gp.covar_module.outputscale.detach().cpu())

                self.gp.eval()
                try:
                    with gpytorch.settings.cholesky_jitter(1e-2):  # bigger than fit jitter
                        output = self.gp(current_train_x)
                        mll_val = float(mll(output, current_train_y.squeeze(-1)).detach().cpu())
                except NotPSDError as e:
                    print(f"  [Warning] NotPSD during MLL eval: {e}")
                    mll_val = float("nan")
            spec = _spectral_snapshot(self.gp.covar_module, current_train_x, max_n=200)

            if self.layout is None:
                base_k = self.gp.covar_module.base_kernel  # DoubleSumSetKernel
                ell_x = float(base_k.ell_x.detach().cpu())
                # Kernel summaries
                D = current_train_x.size(-1)
                m = self.n_points
                d_elem = D // m
                Ek0 = _k0_self_mean(current_train_x, m, d_elem, base_k.ell_x).mean().item()

                # perm invariance check (single-set): permute within set (no vector dims)
                perm_diff = _perm_invariance_check_two_sets(
                    self.gp.covar_module.base_kernel,
                    current_train_x,
                    vec_dim=0,
                    n_inj=self.n_points,
                    n_prod=0,
                    point_dim=d_elem,
                    trials=5,
                    permute_inj=True,
                    permute_prod=False,
                )

                self.train_log.update({"ell_x": ell_x})
                self.diag.update({"E_k0_self_mean": Ek0, "perm_invariance_max_abs_diff": perm_diff})
            else:
                vec_dim = self.layout["vec_dim"]
                n_inj = self.layout["n_inj"]
                n_prod = self.layout["n_prod"]
                point_dim = self.layout["point_dim"]

                inj_k = components["inj"]
                prod_k = components["prod"]

                ell_x = {
                    "inj": float(inj_k.ell_x.detach().cpu()) if inj_k is not None else float("nan"),
                    "prod": float(prod_k.ell_x.detach().cpu()) if prod_k is not None else float("nan"),
                }

                inj_start = vec_dim
                inj_end = inj_start + n_inj * point_dim
                prod_start = inj_end
                prod_end = prod_start + n_prod * point_dim

                X_inj = current_train_x[:, inj_start:inj_end]
                X_prod = current_train_x[:, prod_start:prod_end]

                Ek0 = {
                    "inj": float(_k0_self_mean(X_inj, n_inj, point_dim, inj_k.ell_x).mean().item())
                        if (n_inj > 0 and inj_k is not None) else float("nan"),
                    "prod": float(_k0_self_mean(X_prod, n_prod, point_dim, prod_k.ell_x).mean().item())
                        if (n_prod > 0 and prod_k is not None) else float("nan"),
                }

                perm_diff = _perm_invariance_check_two_sets(
                    self.gp.covar_module.base_kernel,
                    current_train_x,
                    vec_dim=vec_dim,
                    n_inj=n_inj,
                    n_prod=n_prod,
                    point_dim=point_dim,
                    trials=5,
                    permute_inj=True,
                    permute_prod=True,
                )

                self.train_log.update({"ell_x": ell_x})
                self.diag.update({"E_k0_self_mean": Ek0, "perm_invariance_max_abs_diff": perm_diff})

        self.train_log.update({
            "fit_time_sec": fit_time,
            "mll": mll_val,
            "noise": noise,
            "outputscale": outscale,
        })
        self.diag.update({"spectral_snapshot": spec})

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.gp.state_dict(), f"{save_dir}/model.pt")
            with open(f"{save_dir}/ds_diagnostics.json", "w") as f:
                json.dump({"train_log": self.train_log, "diag": self.diag}, f, indent=2)


class SingleTaskGP_DE(_BaseSingleTaskSetGP):
    """
    DE-only set kernel model.
    - If model_args has n_inj/n_prod, uses 2 vectors + 2 sets composite kernel.
    - Else, falls back to the original single-set DE kernel with n_points.
    """
    def __init__(self, model_args: dict, input_dim: int, output_dim: int):
        super().__init__(output_dim)
        self.model_args = dict(model_args)
        self.input_dim = int(input_dim)

        layout = _detect_two_set_layout(model_args)
        self.layout = layout

        if layout is None:
            self.n_points = int(model_args.get("n_points", 5))
        else:
            self.n_points = None

        self.train_log = {
            "fit_time_sec": None,
            "mll": None,
            "noise": None,
            "outputscale": None,
            "ell_x": None,  # scalar or dict
            "ell_h": None,  # scalar or dict
        }
        self.diag = {
            "d2_median_sample": None,              # scalar or dict
            "perm_invariance_max_abs_diff": None,
            "spectral_snapshot": None,
        }

    def fit_and_save(self, train_x: Tensor, train_y: Tensor, save_dir: Optional[str]):
        if self.output_dim > 1:
            raise RuntimeError("SingleTaskGP_DE is single-output only.")
        current_train_x = train_x.squeeze(0) if train_x.dim() == 3 and train_x.shape[0] == 1 else train_x
        if current_train_x.dim() != 2:
            raise RuntimeError(f"train_x shape after squeeze is {current_train_x.shape}, expected (n,d)")
        if current_train_x.size(-1) != self.input_dim:
            raise RuntimeError(f"input_dim mismatch: expected {self.input_dim}, got {current_train_x.size(-1)}")
        current_train_y = _to_2d(train_y)

        components = None
        if self.layout is None:
            base = DeepEmbeddingSetKernel(n_points=self.n_points).to(current_train_x)
        else:
            vec_dim = self.layout["vec_dim"]
            n_inj = self.layout["n_inj"]
            n_prod = self.layout["n_prod"]
            point_dim = self.layout["point_dim"]

            expected = vec_dim + (n_inj + n_prod) * point_dim
            if current_train_x.size(-1) != expected:
                raise RuntimeError(f"Expected input_dim={expected} for vec_dim={vec_dim}, n_inj={n_inj}, n_prod={n_prod}, point_dim={point_dim}, got {current_train_x.size(-1)}")

            base, components = _build_two_set_two_vec_kernel_ds_or_de(
                kind="de",
                vec_dim=vec_dim,
                n_inj=n_inj,
                n_prod=n_prod,
                point_dim=point_dim,
                vec_kernel_name=self.layout["vec_kernel"],
                combine=self.layout["combine"],
            )
            base = base.to(current_train_x)

        covar = ScaleKernel(base, outputscale_prior=GammaPrior(2.0, 0.15))
        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(1.1, 0.05),
            noise_constraint=GreaterThan(1e-4)
        )
        self.gp = botorch.models.SingleTaskGP(
            train_X=current_train_x, train_Y=current_train_y,
            covar_module=covar, outcome_transform=Standardize(m=1),
            likelihood=likelihood
        ).to(current_train_x)

        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp).to(current_train_x)

        t0 = time.time()
        try:
            with gpytorch.settings.cholesky_jitter(1e-4):
                botorch.fit.fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"  [Warning] L-BFGS fit failed: {e}. Falling back to Adam.")
            optimizer = torch.optim.Adam(self.gp.parameters(), lr=0.1)
            self.gp.train(); mll.train()
            for _ in range(100):
                optimizer.zero_grad()
                output = self.gp(current_train_x)
                loss = -mll(output, current_train_y.squeeze())
                if torch.isnan(loss): break
                loss.backward()
                optimizer.step()
        fit_time = time.time() - t0

        with torch.no_grad():
            noise = float(self.gp.likelihood.noise.detach().cpu())
            outscale = float(self.gp.covar_module.outputscale.detach().cpu())

            self.gp.eval()
            output = self.gp(current_train_x)
            mll_val = float(mll(output, current_train_y.squeeze(-1)).detach().cpu())

            spec = _spectral_snapshot(self.gp.covar_module, current_train_x, max_n=200)

            # Diagnostics that depend on DE base params
            if self.layout is None:
                base_k = self.gp.covar_module.base_kernel  # DeepEmbeddingSetKernel
                ell_x = float(base_k.ell_x.detach().cpu())
                ell_h = float(base_k.ell_h.detach().cpu())

                # Sample median of d_E^2 over random pairs
                N, D = current_train_x.shape
                m = self.n_points
                d_elem = D // m
                sets = current_train_x.reshape(N, m, d_elem)
                pairs = [(random.randrange(N), random.randrange(N)) for _ in range(min(64, max(1, N*(N-1)//2)))]
                d2_vals = []
                for i, j in pairs:
                    Si = sets[i].unsqueeze(0)
                    Sj = sets[j].unsqueeze(0)
                    k_xx = _pairwise_rbf(Si, Si, base_k.ell_x).mean()
                    k_yy = _pairwise_rbf(Sj, Sj, base_k.ell_x).mean()
                    k_xy = _pairwise_rbf(Si, Sj, base_k.ell_x).mean()
                    d2_vals.append(float((k_xx + k_yy - 2.0 * k_xy).clamp_min(0.0).item()))
                d2_med = float(torch.tensor(d2_vals).median().item()) if len(d2_vals) > 0 else float("nan")

                perm_diff = _perm_invariance_check_two_sets(
                    self.gp.covar_module.base_kernel,
                    current_train_x,
                    vec_dim=0,
                    n_inj=self.n_points,
                    n_prod=0,
                    point_dim=d_elem,
                    trials=5,
                    permute_inj=True,
                    permute_prod=False,
                )

                self.train_log.update({"ell_x": ell_x, "ell_h": ell_h})
                self.diag.update({"d2_median_sample": d2_med, "perm_invariance_max_abs_diff": perm_diff})

            else:
                vec_dim = self.layout["vec_dim"]
                n_inj = self.layout["n_inj"]
                n_prod = self.layout["n_prod"]
                point_dim = self.layout["point_dim"]

                inj_k = components["inj"]
                prod_k = components["prod"]

                ell_x = {
                    "inj": float(inj_k.ell_x.detach().cpu()) if inj_k is not None else float("nan"),
                    "prod": float(prod_k.ell_x.detach().cpu()) if prod_k is not None else float("nan"),
                }
                ell_h = {
                    "inj": float(inj_k.ell_h.detach().cpu()) if inj_k is not None else float("nan"),
                    "prod": float(prod_k.ell_h.detach().cpu()) if prod_k is not None else float("nan"),
                }

                inj_start = vec_dim
                inj_end = inj_start + n_inj * point_dim
                prod_start = inj_end
                prod_end = prod_start + n_prod * point_dim

                X_inj = current_train_x[:, inj_start:inj_end]
                X_prod = current_train_x[:, prod_start:prod_end]

                def _d2_med_for_block(X_block: Tensor, m: int, d_elem: int, ell_x_block: Tensor) -> float:
                    if m <= 0 or X_block.size(0) < 2:
                        return float("nan")
                    N = X_block.size(0)
                    sets = X_block.reshape(N, m, d_elem)
                    pairs = [(random.randrange(N), random.randrange(N)) for _ in range(min(64, max(1, N*(N-1)//2)))]
                    vals = []
                    for i, j in pairs:
                        Si = sets[i].unsqueeze(0)
                        Sj = sets[j].unsqueeze(0)
                        k_xx = _pairwise_rbf(Si, Si, ell_x_block).mean()
                        k_yy = _pairwise_rbf(Sj, Sj, ell_x_block).mean()
                        k_xy = _pairwise_rbf(Si, Sj, ell_x_block).mean()
                        vals.append(float((k_xx + k_yy - 2.0 * k_xy).clamp_min(0.0).item()))
                    return float(torch.tensor(vals).median().item()) if len(vals) else float("nan")

                d2_med = {
                    "inj": _d2_med_for_block(X_inj, n_inj, point_dim, inj_k.ell_x)
                        if inj_k is not None else float("nan"),
                    "prod": _d2_med_for_block(X_prod, n_prod, point_dim, prod_k.ell_x)
                        if prod_k is not None else float("nan"),
                }

                perm_diff = _perm_invariance_check_two_sets(
                    self.gp.covar_module.base_kernel,
                    current_train_x,
                    vec_dim=vec_dim,
                    n_inj=n_inj,
                    n_prod=n_prod,
                    point_dim=point_dim,
                    trials=5,
                    permute_inj=True,
                    permute_prod=True,
                )

                self.train_log.update({"ell_x": ell_x, "ell_h": ell_h})
                self.diag.update({"d2_median_sample": d2_med, "perm_invariance_max_abs_diff": perm_diff})

        self.train_log.update({
            "fit_time_sec": fit_time,
            "mll": mll_val,
            "noise": noise,
            "outputscale": outscale,
        })
        self.diag.update({"spectral_snapshot": spec})

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.gp.state_dict(), f"{save_dir}/model.pt")
            with open(f"{save_dir}/de_diagnostics.json", "w") as f:
                json.dump({"train_log": self.train_log, "diag": self.diag}, f, indent=2)
