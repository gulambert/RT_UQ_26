import math
import pickle
from pathlib import Path

import numpy as np
import torch
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from sklearn.metrics import r2_score
from tqdm.rich import tqdm

import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

from utils.model import  GP
from utils.acquisition import Jacques
from utils.util import transform_unit_to_marginals
from utils.function import IshigamiBranin


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float64
mean_chunk  = 512

print(f"Running on: {device}")


def _u_to_x(dists, U):
    return transform_unit_to_marginals(U, dists)

def _predict_mean(model, X, chunk=mean_chunk):
    parts = []
    for xb in X.split(chunk):
        with torch.no_grad():
            parts.append(model.predict(xb)["y_mean"])
    return torch.cat(parts, dim=0)

def _compute_q2(model, X_test, Y_test, chunk=mean_chunk):
    Y_pred = _predict_mean(model, X_test, chunk).cpu().numpy()
    Y_true = Y_test.cpu().numpy()
    q2_raw = r2_score(Y_true, Y_pred, multioutput="raw_values")
    q2_avg = float(r2_score(Y_true, Y_pred))
    return q2_raw, q2_avg

def _compute_dgsm(model, X_sample):
    jac_mean = (
        model.grad_posterior(X=X_sample, only_mean=True)
        .squeeze(0).squeeze(0).permute(1, 0, 2)  # (N, p, d)
    )
    return jac_mean.pow(2).mean(0).detach() # (p, d)

def _compute_st(model, dists, N_sobol=2**15, sobol_seed=69, chunk=mean_chunk):
    d_in   = len(dists)
    engine = torch.quasirandom.SobolEngine(dimension=2*d_in, scramble=True, seed=sobol_seed)
    U_2d   = engine.draw(N_sobol, dtype=dtype).to(device)
    A = _u_to_x(dists, U_2d[:, :d_in])
    B = _u_to_x(dists, U_2d[:, d_in:])
    fA = _predict_mean(model, A, chunk).to(device)
    fB = _predict_mean(model, B, chunk).to(device)
    p_out = fA.shape[1]
    ST = torch.zeros(p_out, d_in, dtype=dtype, device=device)
    for i in range(d_in):
        AB_i = A.clone()
        AB_i[:, i] = B[:, i]
        fAB_i = _predict_mean(model, AB_i, chunk).to(device)
        diff_sq = (fA - fAB_i).pow(2)
        for k in range(p_out):
            Vk = torch.cat([fA[:, k], fB[:, k]]).var(unbiased=True)
            if Vk > 1e-12:
                ST[k, i] = 0.5 * diff_sq[:, k].mean() / Vk
    Var_f = torch.cat([fA, fB], dim=0).var(dim=0, unbiased=True)
    V_tot = Var_f.sum()
    ST_agg = (Var_f @ ST) / V_tot if V_tot > 1e-12 else torch.zeros(d_in, dtype=dtype, device=device)
    return ST.detach().cpu().numpy(), ST_agg.detach().cpu().numpy()

def _record(metrics, model, X_test, Y_test, s, t, dists,
            N_sobol=2**15, sobol_seed=69, chunk=mean_chunk):
    q2_raw, q2_avg = _compute_q2(model, X_test, Y_test, chunk)
    metrics["Q2"][s, t] = q2_raw
    metrics["Q2_mean"][s, t] = q2_avg
    nu_raw = _compute_dgsm(model, X_test)
    metrics["DGSM_raw"][s, t] = nu_raw.cpu().numpy()
    ST, ST_agg = _compute_st(model, dists, N_sobol, sobol_seed, chunk)
    metrics["ST"][s, t] = ST
    metrics["ST_agg"][s, t] = ST_agg


N_Xs = 3000
d = 3
p = 2
N_init = 20
N_AL_iter = 50
num_rep = 15
n_record_tot = N_AL_iter + 1
sauv_path = Path("Jacques_MO_IshigamiBranin.pkl")

problem = IshigamiBranin(noise_std_rel=0.05).to(dtype=dtype, device=device)

metrics = {
    "Q2" : np.full((num_rep, n_record_tot, p), np.nan),
    "Q2_mean" : np.full((num_rep, n_record_tot), np.nan),
    "DGSM_raw" : np.full((num_rep, n_record_tot, p, d), np.nan),
    "ST" : np.full((num_rep, n_record_tot, p, d), np.nan),
    "ST_agg" : np.full((num_rep, n_record_tot, d), np.nan),
}

Xtest = (
    draw_sobol_samples(bounds=problem.bounds, n=10_000, q=1, seed=69009)
    .squeeze(1).to(dtype=dtype, device=device)
)
Ytest = problem.evaluate_true(Xtest).to(dtype=dtype, device=device)  # noiseless ✓

Xs = (
    draw_sobol_samples(bounds=problem.bounds, n=N_Xs, q=1, seed=69130)
    .squeeze(1).to(dtype=dtype, device=device)
)

for s, seed in enumerate(tqdm(range(num_rep), desc="seeds")): #AL Loop
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = (
        draw_sobol_samples(bounds=problem.bounds, n=N_init, q=1, seed=seed)
        .squeeze(1).to(dtype=dtype, device=device)
    )
    Y = problem(X)

    model = GP(model_type="mo_indep", noise=True).fit(X, Y)

    _record(metrics, model, Xtest, Ytest, s=s, t=0,
            dists=problem.dists, N_sobol=2**16, sobol_seed=69)

    for i in tqdm(range(N_AL_iter), desc="AL iters", leave=False):

        acqf = Jacques(model=model, global_points=Xs, include_obs_noise=True)

        new_x, _ = optimize_acqf(
            acq_function=acqf,
            q=1,
            bounds=problem.bounds,
            raw_samples=512,
            num_restarts=40,
        )
        new_y = problem(new_x)

        X = torch.cat([X, new_x], dim=0)
        Y = torch.cat([Y, new_y], dim=0)

        model = GP(model_type="mo_indep", noise=True).fit(X, Y, verbose=False)

        _record(metrics, model, Xtest, Ytest, s=s, t=i + 1,
                dists=problem.dists, N_sobol=2**16, sobol_seed=69)

    with open(sauv_path, "wb") as fh:
        pickle.dump({"metrics": metrics, "completed_seeds": s + 1}, fh)
    print(f"[seed {s}] saved ->> {sauv_path}")