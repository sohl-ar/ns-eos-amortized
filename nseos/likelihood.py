"""Likelihood implied *exactly* by the simulator (fix B3) + the TOV interpolation grid.

v1's Dynesty likelihood targeted a different posterior than the network:
  * GW: assumed m1 = m2 from Mc (simulator: m1, m2 ~ iid U(1.1, 2.0)) and ignored Mc noise;
  * NICER: used M_obs as the true mass (ignored sigma_M);
  * pulsar: one-sided soft penalty (simulator: m ~ U(1.4, min(Mmax, 2.5)) + noise);
  * prior: plain box (simulator: box truncated by accept_eos);
  * coarse 10^4 grid with NaN holes filled by neighbour averages, never validated.

Here every term is the marginal of the simulator's generative model:
  GW     : 2/(hi-lo)^2 ∫dMc N(Mc_obs; Mc, sMc) ∫_{q_min(Mc)}^1 dq Mc g(q)^2 N(Lt_obs; Lt(m1, m2), sLt),
           m1 = Mc g(q), m2 = q m1, g(q) = (1+q)^(1/5) q^(-3/5)     [(m1,m2) -> (Mc,q) Jacobian = Mc g^2]
  NICER  : 1/(hi-lo) ∫_lo^hi dm N(M_obs; m, sM) N(R_obs; R(m), sR)
  Pulsar : [Phi((u - m)/s) - Phi((1.4 - m)/s)] / (u - 1.4),  u = min(Mmax, 2.5)
  Prior  : uniform box x 1[accept_eos]
All integrals use Gauss-Legendre in log space. The core is numpy-only (unit-testable).
"""
import math

import numpy as np

from .simulator import SIM_CFG, chirp_mass, interp_lambda, lambda_tilde, stable_branch  # noqa: F401

LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
_GL = {n: np.polynomial.legendre.leggauss(n) for n in (16, 24, 32, 48, 64)}


def _gl_nodes(a, b, n):
    """Gauss-Legendre nodes/weights on [a, b] (a, b may be arrays -> broadcast on last axis)."""
    x, w = _GL[n]
    a, b = np.asarray(a, float)[..., None], np.asarray(b, float)[..., None]
    half = 0.5 * (b - a)
    return a + half * (x + 1.0), half * w


def _logsumexp(a):
    a = np.asarray(a, float)
    m = np.max(a)
    if not np.isfinite(m):
        return -np.inf
    return float(m + np.log(np.sum(np.exp(a - m))))


def _lognorm(x, mu, s):
    return -0.5 * ((x - mu) / s) ** 2 - np.log(s) - LOG_SQRT_2PI


def _log_ndtr(z):
    if z > -30.0:
        return math.log(0.5 * math.erfc(-z / math.sqrt(2.0)))
    return -0.5 * z * z - math.log(-z) - LOG_SQRT_2PI + math.log1p(-1.0 / (z * z) + 3.0 / z ** 4)


def _log_ndtr_diff(b, a):
    """log(Phi(b) - Phi(a)) for b > a, stable in both tails."""
    if b <= a:
        return -np.inf
    if a > 0:  # upper tail: Phi(b)-Phi(a) = Phi(-a)-Phi(-b)
        b, a = -a, -b
    lb, la = _log_ndtr(b), _log_ndtr(a)
    d = la - lb
    if d >= 0:
        return -np.inf
    return lb + math.log1p(-math.exp(d))


def _g(q):
    return (1.0 + q) ** 0.2 * q ** -0.6


def _q_min(Mc, lo, hi, n_iter=50):
    """Smallest q in (0, 1] with m2 = Mc q g(q) >= lo and m1 = Mc g(q) <= hi (vectorised bisection).
    m1 decreases and m2 increases with q, so the support in q is the single interval [q_min, 1]."""
    Mc = np.asarray(Mc, float)
    a, b = np.full_like(Mc, 1e-4), np.ones_like(Mc)
    for _ in range(n_iter):
        mid = 0.5 * (a + b)
        ok = (Mc * mid * _g(mid) >= lo) & (Mc * _g(mid) <= hi)
        b = np.where(ok, mid, b)
        a = np.where(ok, a, mid)
    at1 = Mc * 2.0 ** 0.2
    feasible = (at1 >= lo) & (at1 <= hi)
    return np.where(feasible, b, np.nan)


class CurveEOS:
    """Exact EOS from a stored stable branch, interpolated exactly as the simulator does
    (linear in R, log-linear in Lambda), with the same interface as a grid point."""

    def __init__(self, branch):
        Ms, Rs, Ls = (np.asarray(a, float) for a in branch)
        self.Ms, self.Rs, self.Ls = Ms, Rs, Ls
        self.Mmax, self.Mmin, self.valid = float(Ms[-1]), float(Ms[0]), 1.0

    def R(self, m):
        return np.interp(m, self.Ms, self.Rs)

    def L(self, m):
        return interp_lambda(m, self.Ms, self.Ls)


class _GridPoint:
    def __init__(self, vals, m_grid):
        self.Mmax, self.Mmin, self.valid = float(vals[0]), float(vals[1]), float(vals[2])
        K = len(m_grid)
        self._R, self._logL, self.m_grid = vals[3:3 + K], vals[3 + K:3 + 2 * K], m_grid

    def R(self, m):
        return np.interp(m, self.m_grid, self._R)

    def L(self, m):
        return np.exp(np.interp(m, self.m_grid, self._logL))


def loglike_gw(obs, eos, cfg=SIM_CFG, n_mc=16, n_q=24):
    Lt_obs, Mc_obs = obs["x"]
    sLt, sMc = obs["sigma"]
    lo, hi = cfg["GW_M_LO"], min(eos.Mmax, cfg["GW_M_HI"])
    if hi <= lo:
        return -np.inf
    mc_lo, mc_hi = chirp_mass(lo, lo), chirp_mass(hi, hi)
    a, b = max(mc_lo, Mc_obs - 7 * sMc), min(mc_hi, Mc_obs + 7 * sMc)
    if b <= a:  # observed Mc far outside the support: integrate over the full support in log space
        a, b = mc_lo, mc_hi
    Mc, wMc = _gl_nodes(a, b, n_mc)          # scalar a, b -> shape (n_mc,)
    qmin = _q_min(Mc, lo, hi)
    ok = np.isfinite(qmin) & (qmin < 1.0)
    if not ok.any():
        return -np.inf
    Mc, wMc, qmin = Mc[ok], wMc[ok], qmin[ok]
    q, wq = _gl_nodes(qmin, np.ones_like(qmin), n_q)          # (n_mc, n_q)
    m1 = Mc[:, None] * _g(q)
    m2 = q * m1
    Lt = lambda_tilde(m1, m2, eos.L(m1), eos.L(m2))
    log_int = (np.log(wMc)[:, None] + np.log(wq) + _lognorm(Mc_obs, Mc, sMc)[:, None]
               + np.log(Mc[:, None] * _g(q) ** 2) + _lognorm(Lt_obs, Lt, sLt))
    return _logsumexp(log_int) + math.log(2.0) - 2.0 * math.log(hi - lo)


def loglike_nicer(obs, eos, cfg=SIM_CFG, n_m=48):
    M_obs, R_obs = obs["x"]
    sM, sR = obs["sigma"]
    lo, hi = cfg["NICER_M_LO"], min(eos.Mmax, cfg["NICER_M_HI"])
    if hi <= lo:
        return -np.inf
    a, b = max(lo, M_obs - 7 * sM), min(hi, M_obs + 7 * sM)
    if b <= a:
        a, b = lo, hi
    m, w = _gl_nodes(a, b, n_m)              # shape (n_m,)
    log_int = np.log(w) + _lognorm(M_obs, m, sM) + _lognorm(R_obs, eos.R(m), sR)
    return _logsumexp(log_int) - math.log(hi - lo)


def loglike_pulsar(obs, eos, cfg=SIM_CFG):
    m_obs, s = obs["x"][0], obs["sigma"][0]
    lo, u = cfg["PSR_M_LO"], min(eos.Mmax, cfg["PSR_M_HI"])
    if u <= lo:
        return -np.inf
    return _log_ndtr_diff((u - m_obs) / s, (lo - m_obs) / s) - math.log(u - lo)


_LL = {"gw": loglike_gw, "nicer": loglike_nicer, "pulsar": loglike_pulsar}


def eos_in_support(eos, cfg=SIM_CFG):
    return eos.valid >= 0.5 and eos.Mmax >= cfg["MMAX_MIN"] and eos.Mmin <= cfg["BRANCH_MMIN_MAX"]


def loglike_catalog_eos(catalog, eos, cfg=SIM_CFG):
    if not eos_in_support(eos, cfg):
        return -np.inf
    total = 0.0
    for o in catalog:
        total += _LL[o["type"]](o, eos, cfg)
        if not np.isfinite(total):
            return -np.inf
    return total


# ── TOV interpolation grid ─────────────────────────────────────────────────────
def mass_grid(gcfg):
    return np.round(np.arange(gcfg["m_lo"], gcfg["m_hi"] + 1e-9, gcfg["dm"]), 6)


def _grid_point_values(gc, n_pc, m_grid):
    from .physics import compute_mr_curve
    try:
        curve = compute_mr_curve(list(gc), n_pc=n_pc)
    except Exception:
        curve = None
    K = len(m_grid)
    if curve is None:
        return np.full(3 + 2 * K, np.nan)
    Ms, Rs, Ls = stable_branch(curve)
    if len(Ms) < 3:
        return np.full(3 + 2 * K, np.nan)
    R = np.interp(m_grid, Ms, Rs)                        # flat extension beyond the branch ends
    logL = np.interp(m_grid, Ms, np.log(np.maximum(Ls, 1e-12)))
    return np.concatenate([[Ms[-1], Ms[0], 1.0], R, logL])


def _fill_nans(arr, max_iter=30):
    """Fill NaN cells with the mean of valid neighbours in the 3^4 stencil (repeat)."""
    arr = arr.copy()
    for _ in range(max_iter):
        nan = np.isnan(arr)
        if not nan.any():
            break
        acc, cnt = np.zeros_like(arr), np.zeros_like(arr)
        for shift in np.ndindex(3, 3, 3, 3):
            s = tuple(int(v) - 1 for v in shift)
            if s == (0, 0, 0, 0):
                continue
            rolled = arr
            for ax, d in enumerate(s):
                if d:
                    rolled = np.roll(rolled, d, axis=ax)
                    idx = [slice(None)] * 4
                    idx[ax] = 0 if d == 1 else -1      # do not wrap around
                    rolled = rolled.copy()
                    rolled[tuple(idx)] = np.nan
            v = ~np.isnan(rolled)
            acc[v] += rolled[v]
            cnt[v] += 1
        fill = nan & (cnt > 0)
        arr[fill] = acc[fill] / cnt[fill]
    if np.isnan(arr).any():
        arr[np.isnan(arr)] = np.nanmean(arr)
    return arr


def build_grid(cfg, n_cores=None, partial_path=None, block=1000):
    """TOV grid over the prior box. Resumable: with partial_path, finished blocks of grid points
    are saved as they complete and skipped on restart."""
    import multiprocessing
    import os
    import pickle
    from functools import partial
    from itertools import product
    gcfg, pb = cfg["grid"], cfg["prior_bounds"]
    axes = [np.linspace(pb[f"gamma{k}"][0], pb[f"gamma{k}"][1], n) for k, n in enumerate(gcfg["shape"])]
    m_grid = mass_grid(gcfg)
    pts = list(product(*axes))
    worker = partial(_grid_point_values, n_pc=cfg["physics"]["n_pc"], m_grid=m_grid)
    import time
    done, elapsed = [], 0.0
    if partial_path and os.path.exists(partial_path):
        with open(partial_path, "rb") as fh:
            st = pickle.load(fh)
        done, elapsed = st["done"], st["elapsed"]
        print(f"grid: resuming at point {len(done)} / {len(pts)}")
    t0 = time.time()
    with multiprocessing.Pool(n_cores or multiprocessing.cpu_count()) as pool:
        for i0 in range(len(done), len(pts), block):
            done.extend(pool.map(worker, pts[i0:i0 + block], chunksize=10))
            if partial_path:
                with open(partial_path + ".tmp", "wb") as fh:
                    pickle.dump({"done": done, "elapsed": elapsed + time.time() - t0}, fh)
                os.replace(partial_path + ".tmp", partial_path)
            print(f"grid: {len(done)} / {len(pts)} points", flush=True)
    vals = np.array(done).reshape(*gcfg["shape"], -1)
    tov_ok = ~np.isnan(vals[..., 0])
    for f in range(vals.shape[-1]):
        if f == 2:
            continue
        vals[..., f] = _fill_nans(vals[..., f])
    vals[..., 2] = tov_ok.astype(float)
    return {"axes": axes, "values": vals, "m_grid": m_grid, "shape": list(gcfg["shape"]),
            "n_pc": cfg["physics"]["n_pc"], "frac_tov_ok": float(tov_ok.mean()),
            "build_s": elapsed + time.time() - t0}


class GridEOS:
    def __init__(self, grid):
        from scipy.interpolate import RegularGridInterpolator
        self.m_grid = np.asarray(grid["m_grid"])
        self.interp = RegularGridInterpolator(tuple(grid["axes"]), grid["values"], method="linear",
                                              bounds_error=False, fill_value=None)

    def __call__(self, theta):
        return _GridPoint(self.interp(np.atleast_2d(theta))[0], self.m_grid)

    def batch(self, thetas):
        return self.interp(np.atleast_2d(thetas))


def make_loglike(catalog, grid_eos, cfg=SIM_CFG):
    def ll(theta):
        return loglike_catalog_eos(catalog, grid_eos(theta), cfg)
    return ll


def validate_grid(grid_eos, samples, probe, cfg, m_max_check=2.2):
    """Interpolation error vs exact curves on held-out thetas + truncation-boundary accuracy."""
    m_grid = grid_eos.m_grid
    dR, dlogL, dMmax = [], [], []
    for s in samples:
        Ms, Rs, Ls = s["curve"]
        g = grid_eos(s["theta_eos"])
        m = m_grid[(m_grid >= 1.0) & (m_grid <= min(Ms[-1], m_max_check))]
        dR.append(g.R(m) - np.interp(m, Ms, Rs))
        dlogL.append(np.log(g.L(m)) - np.log(interp_lambda(m, Ms, Ls)))  # simulator convention
        dMmax.append(g.Mmax - Ms[-1])
    dR, dlogL, dMmax = np.concatenate(dR), np.concatenate(dlogL), np.array(dMmax)
    pred = np.array([eos_in_support(grid_eos(t), cfg["sim"]) for t in probe["theta"]])
    true = np.asarray(probe["accepted"])

    def q(a):
        a = np.abs(a)
        return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
                "p99": float(np.percentile(a, 99)), "max": float(a.max())}
    return {"abs_err_R_km": q(dR), "abs_err_logLambda": q(dlogL), "abs_err_Mmax": q(dMmax),
            "frac_R_err_gt_0.1km": float(np.mean(np.abs(dR) > 0.1)),
            "support_accuracy": float(np.mean(pred == true)),
            "support_false_accept": float(np.mean(pred & ~true)),
            "support_false_reject": float(np.mean(~pred & true)),
            "n_theta": len(samples), "n_probe": int(len(true))}
