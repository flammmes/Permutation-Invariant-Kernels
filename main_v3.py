import argparse, json, os, sys, time, traceback
from datetime import datetime
import multiprocessing
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, List, Tuple

import torch
from torch import Tensor

# BoTorch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.utils.transforms import normalize, unnormalize
from botorch.optim import optimize_acqf
from dataclasses import dataclass
import numpy as np
import torch.nn.functional as F

try:
    from scipy.ndimage import distance_transform_edt
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
# your code
from models import (
    SingleTaskGP, SingleTaskGPerm,
    SingleTaskDKL, SingleTaskDKLDeepSets,
    SingleTaskGP_DS, SingleTaskGP_DE,
)
from test_functions import SMH, JHN

# heavy OPM imports kept only where they are really used
from opm.io.ecl import EclFile


@dataclass
class Feasibility:
    # raw grid mask in (H, W) where True = feasible
    mask: np.ndarray
    # signed distance field in raw grid coords (H, W), torch float64 on device
    sdf: torch.Tensor
    # valid points in normalized coords (M, 2) on device
    valid_xy_norm: torch.Tensor
    # interior score for valid points (M,) on device (e.g., sdf at those points, clamped)
    interior_score: torch.Tensor
    # bounds mapping for x,y (raw): min/max in raw grid coords
    xy_min: torch.Tensor  # shape (2,)
    xy_max: torch.Tensor  # shape (2,)

    def bilinear_sdf(self, xy_norm: torch.Tensor) -> torch.Tensor:
        """
        xy_norm: (..., 2) in [0,1]
        returns: (...) signed distance values (differentiable)
        """
        # map to raw grid coords
        xy_raw = xy_norm * (self.xy_max - self.xy_min) + self.xy_min  # (...,2)
        x = xy_raw[..., 0]
        y = xy_raw[..., 1]

        H, W = self.sdf.shape
        x0 = torch.floor(x).clamp(0, W - 1)
        x1 = (x0 + 1).clamp(0, W - 1)
        y0 = torch.floor(y).clamp(0, H - 1)
        y1 = (y0 + 1).clamp(0, H - 1)

        x0l = x0.long(); x1l = x1.long()
        y0l = y0.long(); y1l = y1.long()

        Ia = self.sdf[y0l, x0l]
        Ib = self.sdf[y1l, x0l]
        Ic = self.sdf[y0l, x1l]
        Id = self.sdf[y1l, x1l]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        return wa * Ia + wb * Ib + wc * Ic + wd * Id

    def barrier_penalty(self, X_unit: torch.Tensor, n_inj: int, n_prod: int,
                        margin: float = 1.5, lam: float = 10.0) -> torch.Tensor:
        """
        X_unit shape: (batch, q, d) or (batch, d) in unit cube (normalized inputs)
        Uses only the well-coordinate part.
        penalty is higher when sdf < margin (near boundary/outside).
        """
        if X_unit.dim() == 2:
            X_unit = X_unit.unsqueeze(1)  # (batch,1,d)
        B, q, d = X_unit.shape
        n_wells = n_inj + n_prod
        coords = X_unit[..., 2:]  # (B,q,2*n_wells)
        coords = coords.reshape(B, q, n_wells, 2)
        sdf_vals = self.bilinear_sdf(coords)  # (B,q,n_wells)

        # soft barrier to keep inside + away from boundary:
        # penalty = softplus(margin - sdf)^2
        pen = F.softplus(margin - sdf_vals) ** 2
        # sum wells, average q
        pen = pen.sum(dim=-1).mean(dim=-1)  # (B,)
        return lam * pen

    def round_to_discrete_feasible(self, X_unit_1d: torch.Tensor,
                                   n_inj: int, n_prod: int,
                                   K: int = 64, beta: float = 2.0) -> torch.Tensor:
        """
        Takes a single candidate (d,) in unit cube.
        Rounds wells to discrete feasible cells, preferring interior cells.
        """
        n_wells = n_inj + n_prod
        rates = X_unit_1d[:2]
        coords = X_unit_1d[2:].reshape(n_wells, 2)  # (n_wells,2)

        # For each well, choose among K nearest feasible cells in norm space,
        # scoring by interior_score - beta*dist^2
        out = []
        used = set()

        # valid set (M,2), interior_score (M,)
        V = self.valid_xy_norm
        S = self.interior_score

        for w in range(n_wells):
            c = coords[w]  # (2,)
            d2 = ((V - c) ** 2).sum(dim=-1)  # (M,)
            # take K nearest
            K_eff = min(K, V.shape[0])
            nn_d2, nn_idx = torch.topk(d2, k=K_eff, largest=False)

            best = None
            best_score = None
            for j in nn_idx.tolist():
                xy = V[j]
                key = (float(xy[0].item()), float(xy[1].item()))
                if key in used:
                    continue
                score = S[j] - beta * d2[j]
                if best is None or score > best_score:
                    best = xy
                    best_score = score

            # fallback (shouldn’t happen unless K too small)
            if best is None:
                best = V[nn_idx[0]]
            used.add((float(best[0].item()), float(best[1].item())))
            out.append(best)

        out = torch.stack(out, dim=0).reshape(-1)  # (2*n_wells,)
        return torch.cat([rates, out], dim=0)

def build_feasibility(device: torch.device, dtype=torch.float64) -> Feasibility:
    ### JOHANSEN
    # init = EclFile('test_functions/i.1047/JOHANSEN.INIT')
    # porv_reshaped = init['PORV'].reshape(16, 189, 149)

    # johansen = porv_reshaped[[8,10,12],:,:]
    # valid_johansen_mask = (johansen > 50000).all(axis=0)

    # seal_form = porv_reshaped[7:8,:,:]
    # valid_seal_form_mask = (seal_form > 5).all(axis=0)
    # valid_seal_form_mask[130:,:40] = False
    # valid_seal_form_mask[:78,45:67] = False

    # total_mask = valid_johansen_mask & valid_seal_form_mask
    # total_mask[:,125:] = False 
    #  
    ## SMEAHEIA
    init = EclFile('test_functions/to_restart/GASSNOVA.INIT')
    porv = init['PORV']
    porv_reshaped = porv.reshape(27,253,109)
    valid_mask = (porv_reshaped > 1e-1).all(axis=0)  # valid_mask shape: (153, 109)
    #also mask axis=0 values lower than 176
    valid_mask[219:,:] = False
    total_mask = valid_mask

    # mask is (H,W) in (y,x)
    mask = total_mask.astype(bool)
    H, W = mask.shape

    din  = distance_transform_edt(mask)
    dout = distance_transform_edt(~mask)
    sdf_np = din - dout  # (H,W)

    # valid points raw
    y, x = np.where(mask)
    pts_raw = np.c_[x, y].astype(np.float64)  # (M,2) in (x,y)

    # IMPORTANT: normalize using min/max of VALID locations (matches JHN)
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    xy_min = np.array([x_min, y_min], dtype=np.float64)
    xy_max = np.array([x_max, y_max], dtype=np.float64)

    denom = (xy_max - xy_min)
    denom[denom == 0.0] = 1.0  # safety if degenerate

    pts_norm = (pts_raw - xy_min) / denom

    sdf = torch.tensor(sdf_np, dtype=dtype, device=device)
    V = torch.tensor(pts_norm, dtype=dtype, device=device)

    # interior score at valid points = sdf at those discrete cells (clamp to >=0)
    sdf_at_valid = sdf[torch.tensor(y, device=device), torch.tensor(x, device=device)].clamp_min(0.0)

    return Feasibility(
        mask=mask,
        sdf=sdf,
        valid_xy_norm=V,
        interior_score=sdf_at_valid,
        xy_min=torch.tensor(xy_min, dtype=dtype, device=device),
        xy_max=torch.tensor(xy_max, dtype=dtype, device=device),
    )

def bayes_opt(model, test_fn, cfg, init_x, init_y, save_dir,
              device, model_name, n_inj: int, n_prod: int):

    q = int(cfg["batch_size"])
    output_dim = init_y.shape[-1]
    bounds = test_fn.bounds.to(device, dtype=torch.float64)

    # unit cube bounds for normalized optimization
    d = init_x.shape[-1]
    unit_bounds = torch.stack([torch.zeros(d, device=device), torch.ones(d, device=device)], dim=0)

    # build feasibility once
    feas = build_feasibility(device=device)

    train_x, train_y = init_x.to(device), init_y.to(device)

    for it in range(cfg["n_BO_iters"]):
        print(f"\niteration {it}")

        # Fit on normalized x (unit cube)
        safe_train_y = train_y.clone()
        # Assuming valid JHN results (scaled) are > 0.1 (arbitrary small threshold)
        valid_mask = safe_train_y > 1e-6 
        if valid_mask.any() and (~valid_mask).any():
            min_valid = safe_train_y[valid_mask].min()
            safe_train_y[~valid_mask] = min_valid * 0.9

        # Fit on normalized x (unit cube)
        norm_x = normalize(train_x, bounds)
        # Pass safe_train_y instead of train_y
        model.fit_and_save(norm_x, safe_train_y, save_dir)

        # Construct base acqf
        acqf = construct_acqf_by_model(model_name, model, norm_x, safe_train_y , test_fn)

        # Wrap with feasibility barrier
        margin = cfg.get("feas_margin", 1.5)       # in raw grid distance units
        lam    = cfg.get("feas_lambda", 10.0)

        class PenalizedAcq(torch.nn.Module):
            def __init__(self, base, feas):
                super().__init__()
                self.base = base
                self.feas = feas
            def forward(self, X):
                val = self.base(X)
                pen = self.feas.barrier_penalty(X, n_inj=n_inj, n_prod=n_prod,
                                                margin=margin, lam=lam)
                return val - pen

        acqf_pen = PenalizedAcq(acqf, feas)

        # Optimize in unit cube
        norm_cands, _ = optimize_acqf(
            acq_function=acqf_pen,
            bounds=unit_bounds,
            q=q,
            num_restarts=cfg.get("num_restarts", 15),
            raw_samples=cfg.get("raw_samples", 512),
            options={"batch_limit": cfg.get("batch_limit", 5), "maxiter": cfg.get("maxiter", 200)},
        )  # (1,q,d) in unit cube

        # normalize candidate shape to (q, d) without breaking q=1
        if norm_cands.dim() == 3:
            # sometimes returned as (1, q, d)
            norm_cands = norm_cands.squeeze(0)
        elif norm_cands.dim() == 2:
            # already (q, d)
            pass
        elif norm_cands.dim() == 1:
            # degenerate case: make it (1, d)
            norm_cands = norm_cands.unsqueeze(0)
        else:
            raise RuntimeError(f"Unexpected norm_cands shape: {tuple(norm_cands.shape)}")


        # FINALIZE: discrete + feasible + unique wells
        finalized = []
        for i in range(norm_cands.shape[0]):
            c = norm_cands[i]
            c2 = feas.round_to_discrete_feasible(c, n_inj=n_inj, n_prod=n_prod,
                                                 K=cfg.get("round_K", 64),
                                                 beta=cfg.get("round_beta", 2.0))
            finalized.append(c2.unsqueeze(0))
        norm_cand_for_eval = torch.cat(finalized, dim=0)  # (q,d)

        # Unnormalize to original bounds and eval
        cand_for_eval = unnormalize(norm_cand_for_eval, bounds)     # GPU
        new_y = test_fn(cand_for_eval.detach().cpu())  # CPU
        if new_y.dim() == 0:
            new_y = new_y.view(1, 1)
        elif new_y.dim() == 1:
            new_y = new_y.view(-1, 1)

        new_y = new_y.to(device=device, dtype=torch.float64)  # <<< add this

        train_x = torch.cat([train_x, cand_for_eval])               # keep train_x on GPU
        train_y = torch.cat([train_y, new_y])

        print("best f", train_y.max().item())
    if save_dir:
        final_results_dir = os.path.join(save_dir, "final_BO_results_main")
        os.makedirs(final_results_dir, exist_ok=True)
        torch.save(train_x.cpu(), os.path.join(final_results_dir, "train_x.pt"))
        torch.save(train_y.cpu(), os.path.join(final_results_dir, "train_y.pt"))
    if train_x.numel() == 0: # Handle case where no points were ever evaluated
        print("No points evaluated in BO loop. Returning initial points or empty if no init.")
        if init_x.numel() > 0:
            # This part needs refinement based on how you want to handle "best" from init only
            # For simplicity, let's just return the first init point if single-objective.
            # For multi-objective, this is more complex.
            best_idx_final = 0
            if init_y.numel() > 0:

                best_idx_final = torch.argmax(init_y.squeeze())
                return init_x[best_idx_final], init_y[best_idx_final]
            else: # No init_y, just return first init_x and empty Y
                return init_x[0] if init_x.numel() > 0 else torch.empty(0, device=device), torch.empty(0, device=device)

        else: # No init_x either
            return torch.empty(0, device=device), torch.empty(0, device=device)



    best_idx_final = torch.argmax(train_y.squeeze())
    return train_x[best_idx_final], train_y[best_idx_final]

def initialize_model(name, args, in_dim, out_dim, device):
    if name == "gp":        return SingleTaskGP(args, in_dim, out_dim)
    if name == "gp_perm":   return SingleTaskGPerm(args, in_dim, out_dim)
    if name == "gp_ds":     return SingleTaskGP_DS(args, in_dim, out_dim)
    if name == "gp_de":     return SingleTaskGP_DE(args, in_dim, out_dim)
    if name == "dkl":       return SingleTaskDKL(args, in_dim, out_dim, device)
    if name == "dkl_ds":    return SingleTaskDKLDeepSets(args, in_dim, out_dim, device)
    raise NotImplementedError(name)

def initialize_points(
    fn,
    n_init: int,
    device: torch.device,
    n_inj: int,
    n_prod: int,
    feas: Optional[Feasibility] = None,  # <-- allow None
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if feas is None:
        feas = build_feasibility(device=device)  # cheap enough for you

    bounds = fn.bounds.to(device, dtype=torch.float64)
    d = fn.dim
    n_wells = n_inj + n_prod
    expected_d = 2 + 2 * n_wells
    if d != expected_d:
        raise ValueError(f"fn.dim={d} but expected {expected_d} for n_wells={n_wells}")

    if n_init <= 0:
        X = torch.empty((0, d), device=device, dtype=torch.float64)
        Y = torch.empty((0, fn.num_objectives if fn.num_objectives > 1 else 1),
                        device=device, dtype=torch.float64)
        return X, Y

    # interior weights
    if alpha <= 0:
        weights = None
    else:
        w = (feas.interior_score + 1e-6) ** alpha
        weights = (w / w.sum()).to(device)

    X_unit = torch.empty((n_init, d), device=device, dtype=torch.float64)
    X_unit[:, :2] = torch.rand((n_init, 2), device=device, dtype=torch.float64)

    V = feas.valid_xy_norm
    M = V.shape[0]
    if M < n_wells:
        raise RuntimeError(f"Not enough feasible cells ({M}) for n_wells={n_wells}")

    for i in range(n_init):
        if weights is None:
            idx = torch.randperm(M, device=device)[:n_wells]
        else:
            idx = torch.multinomial(weights, num_samples=n_wells, replacement=False)

        X_unit[i, 2:] = V[idx].reshape(-1)

    X = unnormalize(X_unit, bounds)                   # stays on GPU if bounds on GPU
    Y = fn(X.detach().cpu())                          # evaluate on CPU
    Y = Y.to(device=device, dtype=torch.float64)      # back to GPU for modeling
    return X.to(device), Y

def construct_acqf_by_model(model_name, model, train_x, train_y, fn):
    sampler = StochasticSampler(torch.Size([128]))
    return qLogExpectedImprovement(model, best_f=train_y.max(), sampler=sampler)


def get_test_function(name: str, seed: int, n_inj: int, n_prod: int):
    name = name.lower()
    if name == "oil": return SMH(negate=True, n_inj=n_inj, n_prod=n_prod)
    if name == "jhn": return JHN(negate=True, n_inj=n_inj, n_prod=n_prod)
    raise NotImplementedError(name)

def run_single_trial(trial, cfg, fn, in_dim, out_dim, root, models,
                     gpu, n_inj, n_prod):

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    torch.manual_seed(cfg["seed"] + trial)

    # --- Directory Setup ---
    # Directory for this specific trial
    tdir = os.path.join(root, f"trial_{trial}") 
    os.makedirs(tdir, exist_ok=True) # Create trial directory
    
    # Directory for models *within this trial*
    mdir_for_trial = os.path.join(tdir, "models") # e.g., .../trial_1/models/
    os.makedirs(mdir_for_trial, exist_ok=True) # Create models subdir for this trial
    # --- End Directory Setup ---

    with open(os.path.join(tdir, "stdout.txt"), "w") as log, \
         redirect_stdout(log), redirect_stderr(log):

        init_x, init_y = initialize_points(fn, cfg["n_init_points"], device, n_inj, n_prod)


        for mid, margs in models.items():
            print(f"\n=== model {mid} ===")
            
            # --- Create model-specific directory INSIDE the trial's models dir ---
            model_specific_save_dir = os.path.join(mdir_for_trial, mid) # e.g., .../trial_1/models/gp_perm
            os.makedirs(model_specific_save_dir, exist_ok=True) # Create it!
            # --- End model-specific directory creation ---

            mdl = initialize_model(margs["model"], margs, in_dim, out_dim, device)
            
            # Pass the newly created directory path to bayes_opt
            bx, by = bayes_opt(mdl, fn, cfg, init_x.clone(), init_y.clone(), # Pass clones if needed
                               model_specific_save_dir, device, 
                               margs["model"], n_inj, n_prod)
            
            print("best", by.cpu().numpy())
            del mdl; torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default")
    parser.add_argument("--bg", action="store_true")
    parser.add_argument("-n", "--name")
    args_cli = parser.parse_args()

    cfg_path = f"./config/{args_cli.config}.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    n_inj, n_prod = cfg["n_inj"], cfg["n_prod"]

    stamp = datetime.now().strftime("%y_%m_%d-%H_%M_%S")
    tag   = args_cli.name or args_cli.config
    root  = f"experiment_results/{stamp}_{tag}_{cfg['test_function'].lower()}"
    os.makedirs(root, exist_ok=True)
    json.dump(cfg, open(f"{root}/config.json", "w"), indent=2)

    torch.set_default_dtype(torch.float64)

    fn      = get_test_function(cfg["test_function"], cfg["seed"], n_inj, n_prod)
    in_dim  = fn.dim
    out_dim = fn.num_objectives

    device_id = cfg.get("device_id", 0)
    torch.cuda.set_device(device_id)

    for t in range(1, cfg["n_trials"] + 1):
        try:
            run_single_trial(
                trial=t,
                cfg=cfg,
                fn=fn,
                in_dim=in_dim,
                out_dim=out_dim,
                root=root,
                models=cfg["models"],
                gpu=device_id,
                n_inj=n_inj,
                n_prod=n_prod,
            )
        except Exception as e:
            print(f"Trial {t} failed: {e}")
            traceback.print_exc()

    os.rename(root, root + "_done")
    print("finished")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()