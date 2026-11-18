"""Importance-sampling correction of the amortised posterior with the exact likelihood
(neural importance sampling, cf. Dax et al. 2023).

Proposal: defensive mixture  q(θ) = (1-α) q_flow(θ | catalog) + α U(prior box).
Weights:  w = π(θ) L(θ) / q(θ), with π the uniform box and L the simulator-exact likelihood
(-inf outside the truncated prior support), i.e. *the same target as dynesty*.

Outputs per catalog:
  * sampling efficiency ε = ESS/n  — a per-catalog quality diagnostic of the network;
  * log-evidence  log Z = log mean(w)  with standard error — directly comparable to dynesty;
  * asymptotically exact, weighted posterior samples (the network only proposes).
"""
import math
import multiprocessing
import time

import numpy as np
import torch

from .model import EOSCatalogDataset, collate_fn, model_log_prob, move_to_device
from .reference import _grid_eos
from .likelihood import loglike_catalog_eos


def _ll_chunk(args):
    catalog, grid_path, sim_cfg, thetas = args
    eos = _grid_eos(grid_path)
    return np.array([loglike_catalog_eos(catalog, eos(t), sim_cfg) for t in thetas])


def loglike_many(catalog, thetas, grid_path, sim_cfg, lo, hi, pool=None, n_chunks=64):
    inside = np.all((thetas >= lo) & (thetas <= hi), axis=1)
    out = np.full(len(thetas), -np.inf)
    idx = np.where(inside)[0]
    if len(idx):
        chunks = np.array_split(idx, min(n_chunks, len(idx)))
        jobs = [(catalog, grid_path, sim_cfg, thetas[c]) for c in chunks]
        res = pool.map(_ll_chunk, jobs) if pool is not None else [_ll_chunk(j) for j in jobs]
        for c, r in zip(chunks, res):
            out[c] = r
    return out


def importance_sample(model, catalog, lo, hi, grid_path, sim_cfg, n=20000, n_steps=50, defensive=0.1,
                      seed=0, pool=None):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    dev = model.theta_mu.device
    batch = move_to_device(collate_fn([EOSCatalogDataset([{"theta_eos": [0] * 4, "catalog": catalog}])[0]]), dev)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    rng = np.random.default_rng(seed)
    n_def = int(round(defensive * n))
    t0 = time.perf_counter()
    th_flow = model.sample(batch, n - n_def, n_steps, generator=g)[0].double().cpu().numpy()
    th = np.concatenate([th_flow, rng.uniform(lo, hi, size=(n_def, 4))])
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_sample = time.perf_counter() - t0
    t0 = time.perf_counter()
    lq_flow = model_log_prob(model, batch, torch.tensor(th, dtype=torch.float32, device=dev)[None], n_steps)
    lq_flow = lq_flow[0].double().cpu().numpy()
    t_logq = time.perf_counter() - t0
    log_vol = float(np.sum(np.log(hi - lo)))
    lq = np.logaddexp(math.log1p(-defensive) + lq_flow, math.log(defensive) - log_vol) if defensive > 0 else lq_flow
    t0 = time.perf_counter()
    ll = loglike_many(catalog, th, grid_path, sim_cfg, lo, hi, pool)
    t_ll = time.perf_counter() - t0
    logw = -log_vol + ll - lq
    logw[~np.isfinite(logw)] = -np.inf
    m = np.max(logw)
    w = np.exp(logw - m)
    sw, sw2 = w.sum(), (w ** 2).sum()
    ess = float(sw ** 2 / sw2) if sw2 > 0 else 0.0
    log_z = float(m + math.log(sw / n)) if sw > 0 else -np.inf
    se_log_z = float(np.sqrt(max(np.mean(w ** 2) / np.mean(w) ** 2 - 1.0, 0.0) / n)) if sw > 0 else np.inf
    return {"samples": th, "weights": w / sw if sw > 0 else w, "ess": ess, "efficiency": ess / n,
            "log_z": log_z, "log_z_se": se_log_z, "n": n, "defensive": defensive,
            "frac_flow_outside_support": float(np.mean(~np.isfinite(ll[: n - n_def]))),
            "time_s": {"sample": t_sample, "log_q": t_logq, "likelihood": t_ll,
                       "total": t_sample + t_logq + t_ll}}


def make_pool(n=None):
    return multiprocessing.Pool(n or multiprocessing.cpu_count())
