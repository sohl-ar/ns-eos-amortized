"""Spectral EOS + TOV/tidal solver (ported from the v1 notebook).

Fix B4: ``gamma_crust`` is an explicit argument of ``build_eos`` / ``compute_mr_curve``.
The v1 crust-systematics cell drew a random crust index and never used it.
"""
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .simulator import interp_lambda, stable_branch  # re-export  # noqa: F401

# CGS constants
G_CGS = 6.67430e-8
C_LIGHT = 2.99792458e10
M_SUN_CGS = 1.98892e33
KM_CM = 1.0e5
MEV_FM3_TO_CGS = 1.6022e33
CGS_TO_MEV_FM3 = 1.0 / MEV_FM3_TO_CGS

RHO_NUC = 2.8e14
EPS_NUC_MEV = RHO_NUC * C_LIGHT ** 2 * CGS_TO_MEV_FM3
P_NUC_MEV = 3.0
GAMMA_CRUST = 1.35
P_MAX_MEV = 1500.0
P_MIN_CGS = 1e-4 * MEV_FM3_TO_CGS

LOG_P_NUC = np.log(P_NUC_MEV)
LOG_P_MAX = np.log(P_MAX_MEV)
LOG_P_RANGE = LOG_P_MAX - LOG_P_NUC


def spectral_gamma(P_mev, gc):
    """Gamma(x) = exp(g0 + g1 x + g2 x^2 + g3 x^3), x = log(P/P_nuc)/log(P_max/P_nuc) in [0, 1]."""
    x = np.clip((np.log(P_mev) - LOG_P_NUC) / LOG_P_RANGE, 0, 1)
    return np.exp(gc[0] + gc[1] * x + gc[2] * x ** 2 + gc[3] * x ** 3)


def build_eos(gc, n_pts=300, gamma_crust=None):
    g_crust = GAMMA_CRUST if gamma_crust is None else float(gamma_crust)
    k_crust = P_NUC_MEV / (EPS_NUC_MEV ** g_crust)
    log_P = np.linspace(LOG_P_NUC, LOG_P_MAX, n_pts)
    P = np.exp(log_P)
    eps = np.zeros(n_pts)
    eps[0] = EPS_NUC_MEV
    h = log_P[1] - log_P[0]
    for i in range(n_pts - 1):
        Pi, ei = P[i], eps[i]
        k1 = (ei + Pi) / spectral_gamma(Pi, gc)
        Ph = np.exp(log_P[i] + h / 2)
        Gh = spectral_gamma(Ph, gc)
        k2 = (ei + h / 2 * k1 + Ph) / Gh
        k3 = (ei + h / 2 * k2 + Ph) / Gh
        Pn = P[i + 1]
        k4 = (ei + h * k3 + Pn) / spectral_gamma(Pn, gc)
        eps[i + 1] = ei + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    n_low = 50
    P_low = np.logspace(-8, np.log10(P_NUC_MEV), n_low, endpoint=False)
    eps_low = (P_low / k_crust) ** (1.0 / g_crust)
    P_all = np.concatenate([P_low, P])
    eps_all = np.concatenate([eps_low, eps])
    gam_all = np.concatenate([np.full(n_low, g_crust), np.array([spectral_gamma(p, gc) for p in P])])
    ei = interp1d(np.log(P_all), eps_all, kind="cubic", bounds_error=False, fill_value="extrapolate")
    gi = interp1d(np.log(P_all), gam_all, kind="cubic", bounds_error=False, fill_value="extrapolate")
    return ei, gi


MAX_RHS_EVALS = 30000   # deterministic work limit (v1 used a 0.5 s wall-clock timeout -> load-dependent prior)


def solve_tov_single(Pc_mev, eps_i, gam_i, rtol=1e-6, max_evals=MAX_RHS_EVALS):
    """Integrate TOV + tidal perturbation. Returns (M_sun, R_km, Lambda) or Nones.
    Fix: the integration is abandoned after a fixed number of RHS evaluations instead of a
    wall-clock time, so acceptance no longer depends on machine load."""
    n_evals = [0]
    Pc_cgs = Pc_mev * MEV_FM3_TO_CGS
    ec_cgs = eps_i(np.log(Pc_mev)) * MEV_FM3_TO_CGS
    r0 = 100.0
    m0 = (4.0 / 3.0) * np.pi * r0 ** 3 * ec_cgs / C_LIGHT ** 2
    gc2 = G_CGS / C_LIGHT ** 2
    gc4 = G_CGS / C_LIGHT ** 4

    def rhs(r, s):
        n_evals[0] += 1
        if n_evals[0] > max_evals:
            raise RuntimeError("TOV work limit exceeded")
        P, m, y = s
        if P <= P_MIN_CGS or r < 1.0:
            return [0.0, 0.0, 0.0]
        Pm = max(P * CGS_TO_MEV_FM3, 1e-20)
        eps = eps_i(np.log(Pm)) * MEV_FM3_TO_CGS
        gam = max(gam_i(np.log(Pm)), 1.01)
        cs2 = gam * P / (eps + P)
        if cs2 >= 1.0 or cs2 <= 0.0:
            return [0.0, 0.0, 0.0]  # causality / stability violation -> integration stalls -> rejected
        beta = 2 * G_CGS * m / (r * C_LIGHT ** 2)
        if beta >= 0.99:
            return [0.0, 0.0, 0.0]
        fac = 1.0 - beta
        A = m + 4 * np.pi * r ** 3 * P / C_LIGHT ** 2
        dPdr = -(eps + P) * gc2 * A / (r ** 2 * fac)
        dmdr = 4 * np.pi * r ** 2 * eps / C_LIGHT ** 2
        dnu = gc2 * A / (r ** 2 * fac)
        e2l = 1.0 / fac
        At = e2l * (1 + 4 * np.pi * r ** 2 * (P - eps) * gc4)
        Q = e2l * (4 * np.pi * gc4 * (5 * eps + 9 * P + (eps + P) / cs2) - 6 / r ** 2) - dnu ** 2
        dydr = (-y ** 2 - y * At - r ** 2 * Q) / r
        return [dPdr, dmdr, dydr]

    def surface(r, s):
        return s[0] - P_MIN_CGS
    surface.terminal = True
    surface.direction = -1

    try:
        sol = solve_ivp(rhs, [r0, 50 * KM_CM], [Pc_cgs, m0, 2.0], events=surface, method="RK45",
                        rtol=rtol, atol=1e-9 * rtol / 1e-6, max_step=5000.0)
        if sol.t_events[0].size == 0:
            return None, None, None
        Rcm = sol.t_events[0][0]
        _, Mg, ys = sol.y_events[0][0]
        R, M = Rcm / KM_CM, Mg / M_SUN_CGS
        if M <= 0 or R <= 0 or R > 30:
            return None, None, None
        C = G_CGS * Mg / (Rcm * C_LIGHT ** 2)
        num = 8 / 5 * C ** 5 * (1 - 2 * C) ** 2 * (2 + 2 * C * (ys - 1) - ys)
        den = (2 * C * (6 - 3 * ys + 3 * C * (5 * ys - 8))
               + 4 * C ** 3 * (13 - 11 * ys + C * (3 * ys - 2) + 2 * C ** 2 * (1 + ys))
               + 3 * (1 - 2 * C) ** 2 * (2 - ys + 2 * C * (ys - 1)) * np.log(1 - 2 * C))
        if abs(den) < 1e-30:
            return M, R, None
        Lam = 2 / 3 * (num / den) / C ** 5
        if Lam < 0 or not np.isfinite(Lam):
            return M, R, None
        return M, R, Lam
    except Exception:
        return None, None, None


def compute_mr_curve(gc, n_pc=30, gamma_crust=None, rtol=1e-6, early_stop=True, refine_peak=4,
                     refine_end=8):
    """M-R-Lambda curve, ordered by increasing central pressure (None if < 5 valid stars).

    early_stop  : stop after two consecutive mass decreases past the maximum (only the stable
                  branch is used; a few unstable points are kept).
    refine_peak : golden-section steps on log(Pc) around an *interior* mass maximum.
    refine_end  : bisection steps on log(Pc) between the last valid star and the first rejected
                  one. For many EOS the branch never turns over: it ends where the EOS becomes
                  acausal (c_s^2 >= 1) at the centre, so "M_max" is the mass at that endpoint. On
                  the coarse grid this endpoint was only located to one grid step, biasing M_max
                  low by up to ~0.15 Msun (fix found by the convergence stage)."""
    ei, gi = build_eos(gc, gamma_crust=gamma_crust)

    def star(lpc):
        M, R, L = solve_tov_single(np.exp(lpc), ei, gi, rtol=rtol)
        return None if (M is None or L is None) else (lpc, M, R, L)

    grid = np.linspace(np.log(P_NUC_MEV * 2), np.log(P_MAX_MEV * 0.8), n_pc)
    pts = []   # (log Pc, M, R, Lambda)
    n_down, first_fail_after = 0, None
    for lpc in grid:
        s = star(lpc)
        if s is not None:
            n_down = n_down + 1 if (pts and s[1] < pts[-1][1]) else 0
            pts.append(s)
            first_fail_after = None
            if early_stop and n_down >= 2 and len(pts) >= 5:
                break
        elif pts and first_fail_after is None:
            first_fail_after = lpc
    if len(pts) < 5:
        return None
    i = int(np.argmax([p[1] for p in pts]))
    if refine_end and i == len(pts) - 1 and first_fail_after is not None:
        lo, hi = pts[-1][0], first_fail_after
        for _ in range(refine_end):
            mid = 0.5 * (lo + hi)
            s = star(mid)
            if s is None:
                hi = mid
            else:
                pts.append(s)
                lo = mid
        pts.sort(key=lambda p: p[0])
        i = int(np.argmax([p[1] for p in pts]))
    if refine_peak and 0 < i < len(pts) - 1:
        a, b = pts[i - 1][0], pts[i + 1][0]
        g = (np.sqrt(5) - 1) / 2
        cache = {}

        def mass_at(lpc):
            if lpc not in cache:
                M, R, L = solve_tov_single(np.exp(lpc), ei, gi, rtol=rtol)
                cache[lpc] = (M, R, L)
            return cache[lpc][0] if cache[lpc][0] is not None and cache[lpc][2] is not None else -np.inf
        c, d = b - g * (b - a), a + g * (b - a)
        for _ in range(refine_peak):
            if mass_at(c) > mass_at(d):
                b, d = d, c
                c = b - g * (b - a)
            else:
                a, c = c, d
                d = a + g * (b - a)
        for lpc, (M, R, L) in cache.items():
            if M is not None and L is not None:
                pts.append((lpc, M, R, L))
        pts.sort(key=lambda p: p[0])
    _, Ms, Rs, Ls = (np.array(v) for v in zip(*pts))
    return Ms, Rs, Ls


def branch_for_theta(gc, n_pc=30, gamma_crust=None):
    """Stable branch (Ms, Rs, Ls) as float32 array (3, n), or None (for plotting bands)."""
    try:
        curve = compute_mr_curve(list(gc), n_pc=n_pc, gamma_crust=gamma_crust)
    except Exception:
        return None
    if curve is None:
        return None
    br = stable_branch(curve)
    return np.stack(br).astype(np.float32) if len(br[0]) >= 3 else None


def accepted_branch(gc, n_pc=30, gamma_crust=None, cfg=None):
    """Branch if theta passes accept_eos (with an optional non-default crust), else None."""
    from .simulator import accept_eos, SIM_CFG
    try:
        curve = compute_mr_curve(list(gc), n_pc=n_pc, gamma_crust=gamma_crust)
    except Exception:
        return None
    br, _ = accept_eos(curve, cfg or SIM_CFG)
    return None if br is None else np.stack(br).astype(np.float32)


def derived_observables(gc, n_pc=30, gamma_crust=None, cfg=None):
    """(M_max, R_1.4, Lambda_1.4, inside_training_support) for one theta, or None.
    v1 took the nearest grid point to 1.4 Msun; here we interpolate on the stable branch."""
    from .simulator import accept_eos, SIM_CFG
    try:
        curve = compute_mr_curve(list(gc), n_pc=n_pc, gamma_crust=gamma_crust)
    except Exception:
        return None
    if curve is None:
        return None
    Ms, Rs, Ls = stable_branch(curve)
    if len(Ms) < 3 or Ms[-1] < 1.4 or Ms[0] > 1.4:
        return (float(Ms[-1]) if len(Ms) else np.nan, np.nan, np.nan, False)
    branch, reason = accept_eos(curve, cfg or SIM_CFG)
    return (float(Ms[-1]), float(np.interp(1.4, Ms, Rs)), float(interp_lambda(1.4, Ms, Ls)), reason is None)
