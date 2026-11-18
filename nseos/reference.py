"""Nested-sampling reference posteriors (dynesty), using the simulator-exact likelihood.

v1 went through Bilby; dynesty is called directly here to control seeds and to keep
the *weighted* samples (Kish ESS is reported; v1 compared 10^4 flow samples with a few
hundred resampled nested samples without accounting for that).
"""
import os
import time

import numpy as np

from .io import load_pickle, save_pickle
from .likelihood import GridEOS, loglike_catalog_eos
from .metrics import kish_ess

_GRID_CACHE = {}


def _grid_eos(grid_path):
    if grid_path not in _GRID_CACHE:
        _GRID_CACHE[grid_path] = GridEOS(load_pickle(grid_path))
    return _GRID_CACHE[grid_path]


class _LogLike:
    """Picklable log-likelihood (needed for dynesty checkpoint/restore; closures do not pickle)."""

    def __init__(self, catalog, grid_path, sim_cfg):
        self.catalog, self.grid_path, self.sim_cfg = catalog, grid_path, sim_cfg

    def __call__(self, th):
        v = loglike_catalog_eos(self.catalog, _grid_eos(self.grid_path)(th), self.sim_cfg)
        return v if np.isfinite(v) else -1e300   # dynesty prefers finite values


class _PriorTransform:
    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)

    def __call__(self, u):
        return self.lo + u * (self.hi - self.lo)


def run_dynesty(catalog, grid_path, sim_cfg, lo, hi, seed=0, nlive=500, dlogz=0.1, sample="rwalk",
                bound="multi", maxcall=None, checkpoint_file=None, checkpoint_every=120):
    """Nested sampling with the simulator-exact likelihood. With checkpoint_file, the sampler
    state is saved every `checkpoint_every` seconds and an existing checkpoint is resumed.
    Wall time accumulates over segments (up to the last checkpoint of an interrupted segment)."""
    import json
    import dynesty
    side = checkpoint_file + ".time.json" if checkpoint_file else None
    elapsed_before = 0.0
    if checkpoint_file and os.path.exists(checkpoint_file):
        sampler = dynesty.NestedSampler.restore(checkpoint_file)
        if side and os.path.exists(side):
            with open(side) as fh:
                st = json.load(fh)
            elapsed_before = st["elapsed_before"] + max(0.0, os.path.getmtime(checkpoint_file) - st["segment_start"])
        print(f"dynesty: resuming {os.path.basename(checkpoint_file)}", flush=True)
        resume = True
    else:
        sampler = dynesty.NestedSampler(_LogLike(catalog, grid_path, sim_cfg), _PriorTransform(lo, hi), 4,
                                        nlive=nlive, bound=bound, sample=sample, rstate=np.random.default_rng(seed))
        resume = False
    t0 = time.time()
    if side:
        with open(side, "w") as fh:
            json.dump({"elapsed_before": elapsed_before, "segment_start": t0}, fh)
    kw = {"checkpoint_file": checkpoint_file, "checkpoint_every": checkpoint_every} if checkpoint_file else {}
    if resume:
        sampler.run_nested(resume=True, print_progress=False, **kw)
    else:
        sampler.run_nested(dlogz=dlogz, print_progress=False, maxcall=maxcall, **kw)
    wall = elapsed_before + time.time() - t0
    res = sampler.results
    w = np.exp(res.logwt - res.logz[-1])
    w /= w.sum()
    return {"samples": np.asarray(res.samples), "weights": w, "ess": kish_ess(w),
            "logz": float(res.logz[-1]), "logzerr": float(res.logzerr[-1]),
            "ncall": int(np.sum(res.ncall)), "niter": int(res.niter), "wall_s": wall, "seed": seed,
            "nlive": nlive, "sample": sample, "bound": bound, "dlogz": dlogz, "resumed": resume}


def _job(args):
    out_path, catalog, grid_path, sim_cfg, lo, hi, seed, dcfg = args
    if os.path.exists(out_path):
        return load_pickle(out_path)
    ck = out_path + ".ckpt"
    r = run_dynesty(catalog, grid_path, sim_cfg, lo, hi, seed=seed, nlive=dcfg["nlive"], dlogz=dcfg["dlogz"],
                    sample=dcfg["sample"], bound=dcfg["bound"], maxcall=dcfg.get("maxcall"), checkpoint_file=ck,
                    checkpoint_every=dcfg.get("checkpoint_every", 120))
    save_pickle(out_path, r)
    for f in (ck, ck + ".time.json"):
        if os.path.exists(f):
            os.remove(f)
    return r


def run_many(jobs, n_procs):
    """jobs: list of argument tuples for _job; each result is cached at its own path (resumable)."""
    import multiprocessing
    todo = [j for j in jobs if not os.path.exists(j[0])]
    if todo:
        print(f"dynesty: {len(todo)} runs to do ({len(jobs) - len(todo)} cached) on {n_procs} processes")
        with multiprocessing.Pool(min(n_procs, len(todo))) as pool:
            for _ in pool.imap_unordered(_job, todo):
                pass
    return [load_pickle(j[0]) for j in jobs]


def likelihood_eval_time(catalog, grid_path, sim_cfg, lo, hi, n=2000, seed=0):
    eos = _grid_eos(grid_path)
    rng = np.random.default_rng(seed)
    th = rng.uniform(lo, hi, size=(n, 4))
    t0 = time.perf_counter()
    for t in th:
        loglike_catalog_eos(catalog, eos(t), sim_cfg)
    return (time.perf_counter() - t0) / n
