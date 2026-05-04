from typing import Any, Callable, List, Optional

import botorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors import Posterior
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch import Tensor

from .model import Model

def _to_2d(t: Tensor) -> Tensor:
    t_orig_shape = t.shape
    t = t.squeeze()
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    if t.dim() == 0 and t_orig_shape.numel() > 0 : # Squeezed to scalar but was not empty
        t = t.view(1,1)
    elif t.dim() != 2 or (t.size(-1) != 1 and t.size(0) != 0) : # Allow (0,1) or (N,1)
        if t.numel() == 0 and t.size(-1) !=1 : # Handle empty tensor, e.g. shape (0,0) -> (0,1)
             t = t.view(0,1)
        elif t.numel() > 0 : # Only raise if not empty and still wrong shape
             raise RuntimeError(f"Y shape after squeeze from {t_orig_shape} is {t.shape}, expected (n,1)")
    return t
class SingleTaskGP(Model):

    def __init__(self, model_args, input_dim, output_dim):
        super().__init__()
        self.gp = None
        self.output_dim = output_dim
        self.nu = model_args["nu"] if "nu" in model_args else 2.5

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:  
        return self.gp.posterior(X, output_indices, observation_noise, posterior_transform, **kwargs)

    @property
    def batch_shape(self) -> torch.Size:
        return self.gp.batch_shape

    @property
    def num_outputs(self) -> int:
        return self.gp.num_outputs

    def fit_and_save(self, train_x, train_y, save_dir):
        if self.output_dim > 1:
            raise RuntimeError(
                "SingleTaskGP does not fit tasks with multiple objectives")


        current_train_x = train_x.squeeze(0) if train_x.dim() == 3 and train_x.shape[0] == 1 else train_x
        if current_train_x.dim() != 2:
            raise RuntimeError(f"train_x shape after squeeze is {current_train_x.shape}, expected (n,d)")
        current_train_y = _to_2d(train_y)

        print("train_y shape:", train_y.shape)

        self.gp = botorch.models.SingleTaskGP(
            train_x, train_y, outcome_transform=Standardize(m=1)).to(train_x)
        mll = ExactMarginalLogLikelihood(
            self.gp.likelihood, self.gp).to(train_x)
        fit_gpytorch_mll(mll)


