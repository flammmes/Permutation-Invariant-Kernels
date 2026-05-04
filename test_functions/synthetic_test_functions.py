# test_functions/synthetic_sets.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from botorch.test_functions.synthetic import SyntheticTestFunction


# -------------------------
# Helpers
# -------------------------

def _as_points(X: Tensor, n_points: int) -> Tensor:
    """
    X: (B, 2*n_points) -> (B, n_points, 2)
    """
    B = X.size(0)
    return X.view(B, n_points, 2)


def _rbf_kernel_sqdist(d2: Tensor, sigma: float) -> Tensor:
    # k(x,y) = exp(-||x-y||^2 / (2 sigma^2))
    return torch.exp(-0.5 * d2 / (sigma * sigma + 1e-18))


def _pairwise_sq_dists(A: Tensor, B: Tensor) -> Tensor:
    # A: (..., m, d), B: (..., n, d) -> (..., m, n)
    return torch.cdist(A, B, p=2.0) ** 2


# -------------------------
# Single-set base class
# -------------------------

class _SingleSetBase(SyntheticTestFunction):
    """
    Single unordered set of n_points in 2D.
    Input layout: X = [p1(x,y), ..., pn(x,y)] flattened => dim = 2*n_points
    Default bounds: [-1, 1]^dim
    """
    num_objectives = 1
    _optimal_value = 0.0

    def __init__(self, n_points: int, bounds: Optional[list] = None, noise_std=None, negate: bool = False):
        self.n_points = int(n_points)
        self.dim = 2 * self.n_points
        if bounds is None:
            self._bounds = [(-1.0, 1.0)] * self.dim
        else:
            self._bounds = bounds
        super().__init__(noise_std=noise_std, negate=negate)


# -------------------------
# A.1 Particle Physics (Particle Configuration)
# -------------------------

class ParticlePhysics(_SingleSetBase):
    """
    f(S) = -[ A ||centroid - target||^2 + B sum_{i<j} 1/(alpha dx^2 + beta dy^2 + eps) ]
    """
    def __init__(
        self,
        n_points: int = 10,
        A: float = 1.0,
        B: float = 1.0,
        alpha: float = 1.0,
        beta: float = 2.0,
        eps: float = 1e-4,
        target: Tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ):
        super().__init__(n_points=n_points, **kwargs)
        self.A = float(A)
        self.B = float(B)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.target = torch.tensor(target, dtype=torch.get_default_dtype())

    def evaluate_true(self, X: Tensor) -> Tensor:
        pts = _as_points(X, self.n_points)  # (B,n,2)
        centroid = pts.mean(dim=1)          # (B,2)
        tgt = self.target.to(device=X.device, dtype=X.dtype).view(1, 2)

        attract = self.A * ((centroid - tgt) ** 2).sum(dim=-1)  # (B,)

        # anisotropic repulsion
        dx = pts[:, :, 0].unsqueeze(2) - pts[:, :, 0].unsqueeze(1)  # (B,n,n)
        dy = pts[:, :, 1].unsqueeze(2) - pts[:, :, 1].unsqueeze(1)
        denom = self.alpha * dx * dx + self.beta * dy * dy + self.eps

        # sum over i<j
        inv = 1.0 / denom
        inv = torch.triu(inv, diagonal=1)  # keep i<j
        repulse = self.B * inv.sum(dim=(1, 2))  # (B,)

        f = -(attract + repulse)
        return f.unsqueeze(-1)


# -------------------------
# A.2 Max Area Coverage (Monte Carlo union area + soft overlap penalty)
# -------------------------

class MaxAreaCoverage(_SingleSetBase):
    """
    Places N circles of radius r in [-1,1]^2.
    Objective ~ union area (Monte Carlo) - overlap penalty.
    """
    def __init__(
        self,
        n_points: int = 16,
        r: float = 0.25,
        mc_samples: int = 5000,
        overlap_C: float = 0.2,
        overlap_k: float = 10.0,
        **kwargs,
    ):
        super().__init__(n_points=n_points, **kwargs)
        self.r = float(r)
        self.mc_samples = int(mc_samples)
        self.overlap_C = float(overlap_C)
        self.overlap_k = float(overlap_k)

        # fixed MC points in [-1,1]^2, sampled once (deterministic if you seed globally)
        self.register_buffer("_mc", None, persistent=False)

    def _mc_points(self, device, dtype) -> Tensor:
        if self._mc is None or self._mc.device != device or self._mc.dtype != dtype:
            pts = (2.0 * torch.rand(self.mc_samples, 2, device=device, dtype=dtype) - 1.0)
            self._mc = pts
        return self._mc

    def evaluate_true(self, X: Tensor) -> Tensor:
        B = X.size(0)
        centers = _as_points(X, self.n_points)  # (B,n,2)
        mc = self._mc_points(X.device, X.dtype)  # (M,2)

        # coverage: a mc point is covered if within r of any center
        # d2: (B,M,n)
        d2 = _pairwise_sq_dists(mc.view(1, -1, 2).expand(B, -1, -1), centers)  # (B,M,n)
        covered = (d2 <= (self.r * self.r)).any(dim=-1).to(X.dtype)           # (B,M)
        area_est = covered.mean(dim=1) * 4.0                                  # (B,) area of [-1,1]^2 is 4

        # overlap penalty: sum ReLU(k*(2r - dist)) over pairs
        dist = torch.cdist(centers, centers, p=2.0)  # (B,n,n)
        viol = F.relu(self.overlap_k * (2.0 * self.r - dist))
        viol = torch.triu(viol, diagonal=1)
        overlap_pen = self.overlap_C * viol.sum(dim=(1, 2))

        f = area_est - overlap_pen
        return f.unsqueeze(-1)


# -------------------------
# A.3 Distribution Matching (MMD to fixed target samples)
# -------------------------

class DistributionMatchingMMD(_SingleSetBase):
    """
    f(S) = -max(0, MMD^2_unbiased(S, Y_target))
    """
    def __init__(
        self,
        n_points: int = 20,
        m_target: int = 5000,
        sigma: float = 0.3,
        mix_means: Tuple[Tuple[float, float], Tuple[float, float]] = ((-0.5, -0.5), (0.5, 0.5)),
        mix_std: float = 0.2,
        **kwargs,
    ):
        super().__init__(n_points=n_points, **kwargs)
        self.m_target = int(m_target)
        self.sigma = float(sigma)
        self.mix_std = float(mix_std)
        self.mix_means = torch.tensor(mix_means, dtype=torch.get_default_dtype())  # (2,2)

        self.register_buffer("_Y", None, persistent=False)

    def _target_samples(self, device, dtype) -> Tensor:
        if self._Y is None or self._Y.device != device or self._Y.dtype != dtype:
            means = self.mix_means.to(device=device, dtype=dtype)  # (2,2)
            # sample mixture: choose component 0/1
            comp = torch.randint(0, 2, (self.m_target,), device=device)
            mu = means.index_select(0, comp)  # (M,2)
            Y = mu + self.mix_std * torch.randn(self.m_target, 2, device=device, dtype=dtype)
            self._Y = Y
        return self._Y

    def evaluate_true(self, X: Tensor) -> Tensor:
        pts = _as_points(X, self.n_points)       # (B,n,2)
        Y = self._target_samples(X.device, X.dtype)  # (M,2)

        B, n, _ = pts.shape
        M = Y.size(0)

        # Compute MMD^2 (unbiased) for each batch element
        # K_xx: (B,n,n), K_yy: (M,M), K_xy: (B,n,M)
        d2_xx = _pairwise_sq_dists(pts, pts)
        K_xx = _rbf_kernel_sqdist(d2_xx, self.sigma)

        d2_yy = _pairwise_sq_dists(Y.unsqueeze(0), Y.unsqueeze(0)).squeeze(0)
        K_yy = _rbf_kernel_sqdist(d2_yy, self.sigma)

        d2_xy = _pairwise_sq_dists(pts, Y.unsqueeze(0).expand(B, -1, -1))
        K_xy = _rbf_kernel_sqdist(d2_xy, self.sigma)

        # unbiased: exclude diagonal in xx and yy
        sum_xx = (K_xx.sum(dim=(1, 2)) - K_xx.diagonal(dim1=1, dim2=2).sum(dim=1)) / (n * (n - 1) + 1e-18)
        sum_yy = (K_yy.sum() - torch.diagonal(K_yy).sum()) / (M * (M - 1) + 1e-18)
        sum_xy = K_xy.mean(dim=(1, 2))  # 1/(nM) sum

        mmd2 = sum_xx + sum_yy - 2.0 * sum_xy
        mmd2 = torch.clamp(mmd2, min=0.0)
        f = -mmd2
        return f.unsqueeze(-1)


# -------------------------
# A.4 Max Spanning Tree
# -------------------------

class MaxSpanningTree(_SingleSetBase):
    """
    f(S) = sum_{e in MST_max(S)} w(e), where w(e)=||pi-pj||.
    Uses Prim's algorithm for maximum spanning tree (O(n^2)).
    """
    def __init__(self, n_points: int = 15, **kwargs):
        super().__init__(n_points=n_points, **kwargs)

    def evaluate_true(self, X: Tensor) -> Tensor:
        pts = _as_points(X, self.n_points)  # (B,n,2)
        B, n, _ = pts.shape
        dist = torch.cdist(pts, pts, p=2.0)  # (B,n,n)

        out = []
        for b in range(B):
            D = dist[b]  # (n,n)
            in_tree = torch.zeros(n, device=X.device, dtype=torch.bool)
            key = torch.full((n,), -1e18, device=X.device, dtype=X.dtype)  # max-keys
            key[0] = 0.0
            total = X.new_zeros(())
            for _ in range(n):
                # pick not-in-tree with max key
                masked = key.clone()
                masked[in_tree] = -1e18
                u = torch.argmax(masked)
                in_tree[u] = True
                total = total + key[u]
                # update keys
                key = torch.maximum(key, D[u])
                key[in_tree] = key[in_tree]  # no-op, clarity
            out.append(total)
        return torch.stack(out, dim=0).unsqueeze(-1)


# -------------------------
# A.5 Facility Location (Soft Coverage)
# -------------------------

class FacilityLocationSoftCoverage(_SingleSetBase):
    """
    c(y;S) = 1 - prod_i (1 - exp(-||y-p_i||^2/(2 sigma^2)))
    f(S) = mean_j c(y_j;S) - lambda * repulsion
    """
    def __init__(
        self,
        n_points: int = 12,
        m_clients: int = 2000,
        sigma: float = 0.2,
        rep_lambda: float = 0.02,
        rep_r: float = 0.15,
        rep_kappa: float = 20.0,
        clients_mode: str = "uniform",  # "uniform" or "gmm"
        **kwargs,
    ):
        super().__init__(n_points=n_points, **kwargs)
        self.m_clients = int(m_clients)
        self.sigma = float(sigma)
        self.rep_lambda = float(rep_lambda)
        self.rep_r = float(rep_r)
        self.rep_kappa = float(rep_kappa)
        self.clients_mode = str(clients_mode).lower()
        self.register_buffer("_Y", None, persistent=False)

    def _clients(self, device, dtype) -> Tensor:
        if self._Y is None or self._Y.device != device or self._Y.dtype != dtype:
            if self.clients_mode == "uniform":
                Y = 2.0 * torch.rand(self.m_clients, 2, device=device, dtype=dtype) - 1.0
            elif self.clients_mode == "gmm":
                means = torch.tensor([[-0.5, 0.5], [0.5, -0.5]], device=device, dtype=dtype)
                comp = torch.randint(0, 2, (self.m_clients,), device=device)
                mu = means.index_select(0, comp)
                Y = mu + 0.25 * torch.randn(self.m_clients, 2, device=device, dtype=dtype)
            else:
                raise ValueError(f"Unknown clients_mode={self.clients_mode}")
            self._Y = Y
        return self._Y

    def evaluate_true(self, X: Tensor) -> Tensor:
        centers = _as_points(X, self.n_points)     # (B,n,2)
        Y = self._clients(X.device, X.dtype)       # (M,2)
        B = X.size(0)

        d2 = _pairwise_sq_dists(Y.view(1, -1, 2).expand(B, -1, -1), centers)  # (B,M,n)
        s = _rbf_kernel_sqdist(d2, self.sigma)                                # (B,M,n)
        # c = 1 - prod_i (1 - s_i)
        c = 1.0 - torch.prod(1.0 - s, dim=-1)                                 # (B,M)
        cover = c.mean(dim=1)                                                 # (B,)

        # mild repulsion: softplus(kappa*(r - dist))/kappa over pairs
        dist = torch.cdist(centers, centers, p=2.0)                           # (B,n,n)
        rep = F.softplus(self.rep_kappa * (self.rep_r - dist)) / self.rep_kappa
        rep = torch.triu(rep, diagonal=1).sum(dim=(1, 2))

        f = cover - self.rep_lambda * rep
        return f.unsqueeze(-1)


# -------------------------
# A.6 Soft k-Medoids
# -------------------------

class SoftKMedoids(_SingleSetBase):
    """
    d_tau(y;S) = -tau log sum_i exp(-||y-p_i||^2 / tau)
    f(S) = - mean_j d_tau(y_j;S)
    """
    def __init__(
        self,
        n_points: int = 12,
        m_data: int = 2000,
        tau: float = 0.05,
        data_mode: str = "gmm",  # "gmm", "gaussian", "ring_blob"
        **kwargs,
    ):
        super().__init__(n_points=n_points, **kwargs)
        self.m_data = int(m_data)
        self.tau = float(tau)
        self.data_mode = str(data_mode).lower()
        self.register_buffer("_Y", None, persistent=False)

    def _data(self, device, dtype) -> Tensor:
        if self._Y is None or self._Y.device != device or self._Y.dtype != dtype:
            if self.data_mode == "gaussian":
                Y = 0.6 * torch.randn(self.m_data, 2, device=device, dtype=dtype)
            elif self.data_mode == "gmm":
                means = torch.tensor([[-0.6, -0.6], [0.6, 0.6]], device=device, dtype=dtype)
                comp = torch.randint(0, 2, (self.m_data,), device=device)
                mu = means.index_select(0, comp)
                Y = mu + 0.25 * torch.randn(self.m_data, 2, device=device, dtype=dtype)
            elif self.data_mode == "ring_blob":
                # ring
                t = 2 * math.pi * torch.rand(self.m_data // 2, device=device, dtype=dtype)
                r = 0.7 + 0.05 * torch.randn(self.m_data // 2, device=device, dtype=dtype)
                ring = torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1)
                # blob
                blob = 0.2 * torch.randn(self.m_data - ring.size(0), 2, device=device, dtype=dtype) + torch.tensor([0.0, 0.0], device=device, dtype=dtype)
                Y = torch.cat([ring, blob], dim=0)
            else:
                raise ValueError(f"Unknown data_mode={self.data_mode}")
            self._Y = Y
        return self._Y

    def evaluate_true(self, X: Tensor) -> Tensor:
        centers = _as_points(X, self.n_points)  # (B,n,2)
        Y = self._data(X.device, X.dtype)       # (M,2)
        B = X.size(0)

        d2 = _pairwise_sq_dists(Y.view(1, -1, 2).expand(B, -1, -1), centers)  # (B,M,n)
        tau = max(self.tau, 1e-8)
        softmin = -tau * torch.logsumexp(-d2 / tau, dim=-1)                   # (B,M)
        L = softmin.mean(dim=1)                                               # (B,)
        f = -L
        return f.unsqueeze(-1)


# -------------------------
# Two-set synthetic benchmark (Appendix C style)
# -------------------------

class TwoSetInteraction(SyntheticTestFunction):
    """
    Two-set benchmark: reward producers close to some injector (soft-min),
    add within-set inverse-distance repulsion.
    (Matches Appendix C definition) :contentReference[oaicite:2]{index=2}

    Input layout (flattened):
      X = [ inj_1(x,y), ..., inj_n(x,y), prod_1(x,y), ..., prod_m(x,y) ]
    dim = 2*(n_inj + n_prod)
    """
    _optimal_value = 0.0
    num_objectives = 1

    def __init__(
        self,
        n_inj: int = 4,
        n_prod: int = 6,
        tau: float = 0.05,
        rep_inj: float = 0.05,
        rep_prod: float = 0.05,
        rep_eps: float = 1e-4,
        bounds: list = None,
        noise_std: float = None,
        negate: bool = True,
    ):
        self.n_inj = int(n_inj)
        self.n_prod = int(n_prod)
        self.tau = float(tau)
        self.rep_inj = float(rep_inj)
        self.rep_prod = float(rep_prod)
        self.rep_eps = float(rep_eps)

        self.dim = 2 * (self.n_inj + self.n_prod)
        if bounds is None:
            self._bounds = [(-1.0, 1.0)] * self.dim
        else:
            self._bounds = bounds

        super().__init__(noise_std=noise_std, negate=negate)

    @staticmethod
    def _repulsion_cost(pts: Tensor, w: float, eps: float) -> Tensor:
        if pts.size(1) <= 1 or w <= 0.0:
            return pts.new_zeros(pts.size(0))
        diffs = pts.unsqueeze(2) - pts.unsqueeze(1)      # (B,n,n,2)
        d2 = (diffs ** 2).sum(-1) + eps                  # (B,n,n)
        inv = 1.0 / d2
        eye = torch.eye(pts.size(1), device=pts.device, dtype=pts.dtype).unsqueeze(0)
        inv = inv * (1.0 - eye)
        return w * inv.sum(dim=(1, 2)) / 2.0

    def evaluate_true(self, X: Tensor) -> Tensor:
        B = X.size(0)
        pts = X.view(B, self.n_inj + self.n_prod, 2)
        inj = pts[:, :self.n_inj, :]                     # (B,n_inj,2)
        prod = pts[:, self.n_inj:, :]                    # (B,n_prod,2)

        if self.n_inj > 0 and self.n_prod > 0:
            d2 = ((prod.unsqueeze(2) - inj.unsqueeze(1)) ** 2).sum(-1)  # (B,n_prod,n_inj)
            tau = max(self.tau, 1e-8)
            softmin = -tau * torch.logsumexp(-d2 / tau, dim=2)          # (B,n_prod)
            interaction_cost = softmin.mean(dim=1)                      # (B,)
        else:
            interaction_cost = X.new_zeros(B)

        rep_cost = self._repulsion_cost(inj, self.rep_inj, self.rep_eps) \
                 + self._repulsion_cost(prod, self.rep_prod, self.rep_eps)

        total_cost = interaction_cost + rep_cost
        return total_cost.unsqueeze(-1)