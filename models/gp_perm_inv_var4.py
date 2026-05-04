
from typing import Any, Callable, List, Optional

import math
import os
import time

import numpy as np
import torch
from torch import Tensor

import botorch
from botorch.posteriors import Posterior
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize

import gpytorch
from gpytorch.kernels import Kernel, ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior, Prior
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.utils.errors import NotPSDError
from gpytorch.distributions import MultivariateNormal

from linear_operator.operators import DenseLinearOperator
from linear_operator.utils.errors import NanError
from types import MethodType


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
            raise RuntimeError(
                f"Y shape after squeeze from {t_orig_shape} is {t.shape}, expected (n,1)"
            )
    return t


@torch.jit.script
def _sinkhorn(cost: Tensor, eps: float, iters: int = 150, tol: float = 1e-6) -> Tensor:
    # cost: (B, m, m) with uniform marginals a=b=1/m
    B, m, _ = cost.shape
    logK = -cost / eps
    log_a = -math.log(float(m))
    log_b = log_a
    f = cost.new_zeros(B, m)  # row potentials
    g = cost.new_zeros(B, m)  # col potentials

    for _ in range(iters):
        f_old, g_old = f, g
        f = eps * (log_a - torch.logsumexp(logK + g.unsqueeze(-2) / eps, dim=-1))
        g = eps * (log_b - torch.logsumexp(logK.transpose(-1, -2) + f.unsqueeze(-2) / eps, dim=-1))
        if torch.max((f - f_old).abs().amax(-1), (g - g_old).abs().amax(-1)).max() < tol:
            break

    P_log = logK + (f.unsqueeze(-1) + g.unsqueeze(-2)) / eps
    P = torch.exp(P_log)
    return (P * cost).sum((-2, -1))


class SinkhornMaternKernel(Kernel):
    """
    ANOVA-style engineered kernel:

      D^2 = (Δrate_inj / ell_rate_inj)^2
          + (Δrate_prod / ell_rate_prod)^2
          + ( sqrt(S_inj) / ell_sink_inj )^2
          + ( sqrt(S_prod) / ell_sink_prod )^2

    where S_* is Sinkhorn divergence between the corresponding sets.
    """

    has_lengthscale = False  # we manage multiple lengthscales manually

    def __init__(
        self,
        n_inj: int,
        n_prod: int,
        eps_sink: float = 1e-1,
        nu: float = 2.5,
        lower_lengthscale_bound: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_inj = int(n_inj)
        self.n_prod = int(n_prod)
        self.nu = float(nu)

        self.register_buffer("eps_sink", torch.tensor(float(eps_sink), dtype=torch.float64))
        # Separate robust scales for each set-block (helps a lot in practice)
        self.register_buffer("s_scale_inj", torch.tensor(1.0, dtype=torch.float64))
        self.register_buffer("s_scale_prod", torch.tensor(1.0, dtype=torch.float64))

        param_dtype = torch.float64
        # Four separate lengthscales (2 scalar + 2 set)
        self.register_parameter(
            "raw_rate_inj_lengthscale",
            torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, dtype=param_dtype)),
        )
        self.register_parameter(
            "raw_rate_prod_lengthscale",
            torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, dtype=param_dtype)),
        )
        self.register_parameter(
            "raw_sink_inj_lengthscale",
            torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, dtype=param_dtype)),
        )
        self.register_parameter(
            "raw_sink_prod_lengthscale",
            torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, dtype=param_dtype)),
        )

        c = GreaterThan(lower_lengthscale_bound)
        self.register_constraint("raw_rate_inj_lengthscale", c)
        self.register_constraint("raw_rate_prod_lengthscale", c)
        self.register_constraint("raw_sink_inj_lengthscale", c)
        self.register_constraint("raw_sink_prod_lengthscale", c)
        self.register_buffer("s_scale_ip", torch.tensor(1.0, dtype=torch.float64))

        self.register_parameter(
            "raw_sink_ip_lengthscale",
            torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, dtype=param_dtype)),
        )
        self.register_constraint("raw_sink_ip_lengthscale", c)

    @property
    def sink_ip_lengthscale(self) -> Tensor:
        return self.raw_sink_ip_lengthscale_constraint.transform(self.raw_sink_ip_lengthscale)

    @property
    def rate_inj_lengthscale(self) -> Tensor:
        return self.raw_rate_inj_lengthscale_constraint.transform(self.raw_rate_inj_lengthscale)

    @property
    def rate_prod_lengthscale(self) -> Tensor:
        return self.raw_rate_prod_lengthscale_constraint.transform(self.raw_rate_prod_lengthscale)

    @property
    def sink_inj_lengthscale(self) -> Tensor:
        return self.raw_sink_inj_lengthscale_constraint.transform(self.raw_sink_inj_lengthscale)

    @property
    def sink_prod_lengthscale(self) -> Tensor:
        return self.raw_sink_prod_lengthscale_constraint.transform(self.raw_sink_prod_lengthscale)

    @staticmethod
    def _center_sets(x: Tensor) -> Tensor:
        # x: (..., m, 2)
        return x - x.mean(dim=-2, keepdim=True)

    def _self_ot_diag(self, x: Tensor, m: int) -> Tensor:
        # x: (B, n, m, 2) -> returns (B, n)
        #x = self._center_sets(x)
        x_i = x.unsqueeze(-3).double()  # (B,n,1,m,2)
        x_j = x.unsqueeze(-2).double()  # (B,n,m,1,2)
        C_sq = (x_i - x_j).pow(2).sum(-1)  # (B,n,m,m)
        C = torch.sqrt(C_sq + 1e-12)
        C_flat = C.reshape(-1, m, m)
        W_flat = _sinkhorn(C_flat, float(self.eps_sink.item()), iters=100, tol=1e-4)
        return W_flat.view(x.shape[0], x.shape[1]).to(x.dtype)

    def _ot(self, xa: Tensor, xb: Tensor, m: int) -> Tensor:
        # xa: (B,n1,m,2), xb: (B,n2,m,2) -> (B,n1,n2)
        if m == 0:
            return torch.zeros(xa.size(0), xa.size(1), xb.size(1), device=xa.device, dtype=xa.dtype)

        # xa = self._center_sets(xa)
        # xb = self._center_sets(xb)

        xa_exp = xa.unsqueeze(2).unsqueeze(-2)  # (B,n1,1,m,1,2)
        xb_exp = xb.unsqueeze(1).unsqueeze(-3)  # (B,1,n2,1,m,2)
        diff = xa_exp - xb_exp                  # (B,n1,n2,m,m,2)
        C_sq = (diff ** 2).sum(-1)              # (B,n1,n2,m,m)
        C = torch.sqrt(C_sq + 1e-9)
        C = torch.nan_to_num(C, posinf=1e6)

        C_flat = C.reshape(-1, m, m)  # (B*n1*n2,m,m)
        W_flat = _sinkhorn(C_flat, float(self.eps_sink.item()), iters=100, tol=1e-4)
        return W_flat.view(xa.size(0), xa.size(1), xb.size(1))

    def _sinkhorn_divergence_matrix(self, x1_sets: Tensor, x2_sets: Tensor, m: int, which: str) -> Tensor:
        # x1_sets: (B,n1,m,2), x2_sets: (B,n2,m,2) -> S: (B,n1,n2)
        if m == 0:
            return torch.zeros(x1_sets.size(0), x1_sets.size(1), x2_sets.size(1),
                               device=x1_sets.device, dtype=x1_sets.dtype)

        W12 = self._ot(x1_sets, x2_sets, m)        # (B,n1,n2)
        s11 = self._self_ot_diag(x1_sets, m)       # (B,n1)
        s22 = self._self_ot_diag(x2_sets, m)       # (B,n2)

        S = W12 - 0.5 * s11.unsqueeze(-1) - 0.5 * s22.unsqueeze(-2)
        S = S.clamp_min(0.0)

        if which == "inj":
            scale = self.s_scale_inj
        elif which == "prod":
            scale = self.s_scale_prod
        elif which == "ip":
            scale = self.s_scale_ip
        else:
            raise RuntimeError(f"Unknown which={which}")
        S_fin = S[torch.isfinite(S)]
        if self.training and S_fin.numel() > 0:
            with torch.no_grad():
                med = S_fin.median()
                new_scale = (med + 1e-12).to(scale.dtype)
                # in-place EMA so the buffer stays registered
                scale.mul_(0.9).add_(0.1 * new_scale)

        return S / (scale + 1e-12)

    def _broadcast_ls(self, ls: Tensor, target: Tensor) -> Tensor:
        # ls: (*batch,1) -> broadcast to target dims
        ls = ls.squeeze(-1)
        while ls.dim() < target.dim():
            ls = ls.unsqueeze(-1)
        return ls

    def _compute_distance(self, x1: Tensor, x2: Tensor, diag: bool = False) -> Tensor:
        if diag:
            return torch.zeros(*x1.shape[:-1], dtype=x1.dtype, device=x1.device)

        orig_x1_dim, orig_x2_dim = x1.dim(), x2.dim()
        if x1.dim() == 2:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 2:
            x2 = x2.unsqueeze(0)

        if x1.size(0) != x2.size(0):
            if x1.size(0) == 1:
                x1 = x1.expand(x2.size(0), -1, -1)
            elif x2.size(0) == 1:
                x2 = x2.expand(x1.size(0), -1, -1)
            else:
                raise RuntimeError(f"Incompatible batch sizes: {x1.shape} vs {x2.shape}")

        B, n1, d = x1.shape
        n2 = x2.size(1)

        # --- rates (2 scalar dims) ---
        ls_r_inj = self._broadcast_ls(self.rate_inj_lengthscale, x1.new_zeros(B, n1, n2))
        ls_r_pro = self._broadcast_ls(self.rate_prod_lengthscale, x1.new_zeros(B, n1, n2))

        r_inj_1 = x1[..., 0].unsqueeze(-1)   # (B,n1,1)
        r_inj_2 = x2[..., 0].unsqueeze(-2)   # (B,1,n2)
        dq_inj = (r_inj_1 - r_inj_2) / ls_r_inj.clamp_min(1e-8)  # (B,n1,n2)

        r_pro_1 = x1[..., 1].unsqueeze(-1)
        r_pro_2 = x2[..., 1].unsqueeze(-2)
        dq_pro = (r_pro_1 - r_pro_2) / ls_r_pro.clamp_min(1e-8)  # (B,n1,n2)

        # --- sets ---
        start_inj = 2
        end_inj = start_inj + 2 * self.n_inj
        start_pro = end_inj
        end_pro = start_pro + 2 * self.n_prod

        if end_pro > d:
            raise RuntimeError(
                f"Input dim={d} too small for n_inj={self.n_inj}, n_prod={self.n_prod}. "
                f"Need at least {2 + 2*self.n_inj + 2*self.n_prod}."
            )

        if self.n_inj > 0:
            x1_inj = x1[..., start_inj:end_inj].reshape(B, n1, self.n_inj, 2)
            x2_inj = x2[..., start_inj:end_inj].reshape(B, n2, self.n_inj, 2)
            S_inj = self._sinkhorn_divergence_matrix(x1_inj, x2_inj, self.n_inj, which="inj")
            ls_s_inj = self._broadcast_ls(self.sink_inj_lengthscale, S_inj)
            D_inj = torch.sqrt(S_inj + 1e-12) / ls_s_inj.clamp_min(1e-8)
        else:
            D_inj = torch.zeros(B, n1, n2, device=x1.device, dtype=x1.dtype)

        if self.n_prod > 0:
            x1_pro = x1[..., start_pro:end_pro].reshape(B, n1, self.n_prod, 2)
            x2_pro = x2[..., start_pro:end_pro].reshape(B, n2, self.n_prod, 2)
            S_pro = self._sinkhorn_divergence_matrix(x1_pro, x2_pro, self.n_prod, which="prod")
            ls_s_pro = self._broadcast_ls(self.sink_prod_lengthscale, S_pro)
            D_pro = torch.sqrt(S_pro + 1e-12) / ls_s_pro.clamp_min(1e-8)
        else:
            D_pro = torch.zeros(B, n1, n2, device=x1.device, dtype=x1.dtype)

        # --- injector-producer relational set term ---
        # Build relational sets R^x = {p - i} as an unordered set of size n_inj * n_prod
        if self.n_inj > 0 and self.n_prod > 0:
            # x1_inj: (B,n1,n_inj,2), x1_pro: (B,n1,n_prod,2)
            # Relational displacements: (B,n1,n_inj,n_prod,2)
            R1 = (x1_pro.unsqueeze(-3) - x1_inj.unsqueeze(-2)).reshape(B, n1, self.n_inj * self.n_prod, 2)
            R2 = (x2_pro.unsqueeze(-3) - x2_inj.unsqueeze(-2)).reshape(B, n2, self.n_inj * self.n_prod, 2)

            m_ip = self.n_inj * self.n_prod
            S_ip = self._sinkhorn_divergence_matrix(R1, R2, m_ip, which="ip")

            S_ip = S_ip / math.sqrt(m_ip)

            ls_ip = self._broadcast_ls(self.sink_ip_lengthscale, S_ip)
            D_ip = torch.sqrt(S_ip + 1e-12) / ls_ip.clamp_min(1e-8)
        else:
            D_ip = torch.zeros(B, n1, n2, device=x1.device, dtype=x1.dtype)



        r2 = dq_inj.pow(2) + dq_pro.pow(2) + D_inj.pow(2) + D_pro.pow(2) +  D_ip.pow(2)
        D = torch.sqrt(r2.clamp_min(0.0) + 1e-12)

        if orig_x1_dim == 2 and orig_x2_dim == 2 and D.size(0) == 1:
            D = D.squeeze(0)
        return D

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        if diag:
            return torch.ones(*x1.shape[:-1], dtype=x1.dtype, device=x1.device)

        D = self._compute_distance(x1, x2, diag=False)

        if self.nu == 0.5:
            return torch.exp(-D)
        elif self.nu == 1.5:
            sqrt3_D = math.sqrt(3.0) * D
            return (1.0 + sqrt3_D) * torch.exp(-sqrt3_D)
        elif self.nu == 2.5:
            sqrt5_D = math.sqrt(5.0) * D
            return (1.0 + sqrt5_D + (5.0 / 3.0) * D.pow(2)) * torch.exp(-sqrt5_D)
        else:
            raise RuntimeError("Unsupported nu value. Use 0.5, 1.5, or 2.5.")


class SingleTaskGPerm(botorch.models.model.Model):
    def __init__(self, model_args: dict, input_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

        self.n_inj = model_args.get("n_inj", 3)
        self.n_prod = model_args.get("n_prod", 5)

        self.kernel_nu = model_args.get("nu", 2.5)
        self.kernel_eps_sink = model_args.get("kernel_eps_sink", 1e-1)
        self.kernel_lower_ls_bound = model_args.get("lower_lengthscale_bound", 1e-5)

        self.adam_lr = model_args.get("adam_lr", 0.01)
        self.adam_epochs = model_args.get("adam_epochs", 300)
        self.adam_weight_decay = model_args.get("adam_weight_decay", 0.01)

        self.gp = None

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        if self.gp is None:
            raise RuntimeError("GP is not fit yet.")
        return self.gp.posterior(X, output_indices, observation_noise, posterior_transform, **kwargs)

    @property
    def batch_shape(self):
        if self.gp is None:
            return torch.Size()
        return self.gp.batch_shape

    @property
    def num_outputs(self) -> int:
        if self.gp is None:
            return self.output_dim
        return self.gp.num_outputs

    def fit_and_save(self, train_x: Tensor, train_y: Tensor, save_dir: Optional[str]):
        if self.output_dim > 1:
            raise RuntimeError("SingleTaskGPerm is single-output only.")

        current_train_x = train_x.squeeze(0) if train_x.dim() == 3 and train_x.shape[0] == 1 else train_x
        if current_train_x.dim() != 2:
            raise RuntimeError(f"train_x shape after squeeze is {current_train_x.shape}, expected (n,d)")
        current_train_y = _to_2d(train_y)

        fit_successful = False
        last_exception = None

        try:
            base_kernel = SinkhornMaternKernel(
                n_inj=self.n_inj,
                n_prod=self.n_prod,
                nu=self.kernel_nu,
                eps_sink=self.kernel_eps_sink,
                lower_lengthscale_bound=self.kernel_lower_ls_bound,
            ).to(current_train_x)

            covar_module = ScaleKernel(base_kernel, outputscale_prior=GammaPrior(2.0, 0.15))
            likelihood = GaussianLikelihood(
                noise_prior=GammaPrior(1.1, 0.05),
                noise_constraint=GreaterThan(1e-6),
            )

            jitter_levels = np.logspace(-5, -1, num=5, base=10)
            start_fit_time = time.time()

            for jitter_val in jitter_levels:
                print(f"  Trying with jitter level: {jitter_val:.1E}")
                self.gp = botorch.models.SingleTaskGP(
                    train_X=current_train_x,
                    train_Y=current_train_y,
                    covar_module=covar_module,
                    outcome_transform=Standardize(m=1),
                    likelihood=likelihood,
                ).to(current_train_x)

                mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp).to(current_train_x)

                try:
                    with gpytorch.settings.cholesky_jitter(float(jitter_val)):
                        fit_gpytorch_mll(mll)

                    print(f"  [GP Fit] SUCCESS with Sinkhorn-divergence ANOVA kernel (jitter {jitter_val:.1E}).")
                    fit_successful = True
                    last_exception = None
                    break

                except NanError as e_nan:
                    last_exception = e_nan
                    print(f"    [GP Fit] NaN/Inf loss at jitter {jitter_val:.1E}, retrying...")

                except Exception as e_other:
                    last_exception = e_other
                    print(f"    [GP Fit] Failed at jitter {jitter_val:.1E}: {e_other}")

            print(f"Sinkhorn-divergence ANOVA fitting took {time.time() - start_fit_time:.2f}s.")
            if not fit_successful:
                raise last_exception if last_exception is not None else RuntimeError("Unknown fit failure.")

        except Exception as e_sinkhorn_fit:
            print(f"\n[Fallback Triggered] ANOVA Sinkhorn-divergence kernel FAILED: {e_sinkhorn_fit}")
            print("Attempting fallback to standard Matern kernel with fit_gpytorch_mll (LBFGS)...")
            try:
                fallback_base_kernel = MaternKernel(
                    nu=2.5,
                    ard_num_dims=current_train_x.shape[-1] if current_train_x.shape[-1] > 0 else None,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ).to(current_train_x)
                fallback_covar_module = ScaleKernel(fallback_base_kernel, outputscale_prior=GammaPrior(2.0, 0.15))
                self.gp = botorch.models.SingleTaskGP(
                    train_X=current_train_x,
                    train_Y=current_train_y,
                    covar_module=fallback_covar_module,
                    outcome_transform=Standardize(m=1),
                ).to(current_train_x)
                fallback_mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp).to(current_train_x)
                fit_gpytorch_mll(fallback_mll, max_retries=5)
                print("[Fallback] Training completed with standard Matern kernel.")
            except Exception as e_fallback_fit:
                print(f"[Fallback] ALSO FAILED: {e_fallback_fit}")
                print("!!! CRITICAL: All fitting attempts failed. GP may be unusable. !!!")

        if self.gp is not None and save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, "model.pt")
            torch.save(self.gp.state_dict(), model_path)
            print(f"  [GP-Perm] Saved trained model state_dict to {model_path}")

        # PSD-repair patch (as before)
        if self.gp is not None:
            def _safe_posterior(self_gp_instance, X_post, *a_post, **kw_post):
                try:
                    return super(type(self_gp_instance), self_gp_instance).posterior(X_post, *a_post, **kw_post)
                except NotPSDError:
                    with torch.no_grad():
                        K_lazy = self_gp_instance.covar_module(X_post)
                        K_dense = K_lazy.evaluate()
                        if not isinstance(K_dense, torch.Tensor):
                            K_dense = K_dense.to_dense()
                        eigval, eigvec = torch.linalg.eigh(K_dense)
                        K_spd = eigvec @ torch.diag_embed(eigval.clamp_min(1e-6)) @ eigvec.mT
                    mean_val = self_gp_instance.mean_module(X_post)
                    return MultivariateNormal(mean_val, DenseLinearOperator(K_spd))

            self.gp.posterior = MethodType(_safe_posterior, self.gp)
