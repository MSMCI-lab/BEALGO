#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Insurance pricing table from BE-derived loss distribution (L in {3,4,5}).

- Imports BE algorithms from be_joint_pmf.py
- Severities:
    CI/CD, IoT, FUS, ERP, SCADA ~ Lognormal(mu, sigma) with mu = ln(median)
    SAMM, AD                    ~ Weibull(k, lambda)
- Computes E[S], SD(S), GMD, VaR_0.99, TVaR_0.99 via Laplace inversion
  with correct handling of F0 = P{S=0}, then premium principles.

Outputs a pretty table and a CSV "pricing_vs_L.csv".
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, FrozenSet, Callable, List
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from mpmath import gamma as mp_gamma
from math import pi

# ---- import BE module (Algorithms 1–2) ----
from be_joint_pmf import joint_pmf_by_types

# -----------------------------
# Network & entry parameters
# -----------------------------
V = list(range(1, 8))
node_types = {
    1: "CI/CD",
    2: "FUS",
    3: "IoT",
    4: "SAMM",
    5: "ERP",
    6: "AD",
    7: "SCADA",
}

# Corrected Δ (8 edges)
DELTA_EDGES: Dict[Tuple[int,int], float] = {
    (1,2): 0.10,
    (2,3): 0.10,
    (3,4): 0.30,
    (4,7): 0.10,
    (5,6): 0.20,
    (6,7): 0.15,
    (1,4): 0.15,
    (4,6): 0.60,
}

# Direct entry at nodes 1 and 5
alpha1, alpha5 = 0.12, 0.08
DIRECT_LAW: Dict[FrozenSet[int], float] = {
    frozenset(): (1 - alpha1) * (1 - alpha5),
    frozenset({1}): alpha1 * (1 - alpha5),
    frozenset({5}): (1 - alpha1) * alpha5,
    frozenset({1, 5}): alpha1 * alpha5,
}

# -----------------------------
# Severities (exact per user)
# -----------------------------
# Lognormal(mu, sigma) where mu = ln(median)
SEV = {
    "CI/CD":  ("lognorm", np.log(30_000.0), 0.6),
    "IoT":    ("lognorm", np.log(25_000.0), 0.6),
    "FUS":    ("lognorm", np.log(40_000.0), 0.7),
    "ERP":    ("lognorm", np.log(80_000.0), 0.9),
    "SCADA":  ("lognorm", np.log(150_000.0), 1.0),
    "SAMM":   ("weibull", 0.8,  90_000.0),   # (k, lambda)
    "AD":     ("weibull", 0.9, 110_000.0),
}

# Quadrature nodes for LSTs
_HERM_N = 64
_HX, _HW = hermgauss(_HERM_N)      # for lognormal (Gauss–Hermite)
_LAG_N = 64
_LX, _LW = laggauss(_LAG_N)        # for Weibull   (Gauss–Laguerre)

# ---------- LSTs ----------
def phi_lognorm(s: complex, mu: float, sigma: float) -> complex:
    """
    Correct Gauss–Hermite rule:
      E[f(Z)] = (1/sqrt(pi)) * sum_i w_i * f(sqrt(2)*x_i),  Z~N(0,1)
    Here f(z) = exp(-s * exp(mu + sigma*z)).
    """
    z = np.sqrt(2.0) * _HX
    vals = np.exp(-s * np.exp(mu + sigma * z))
    return (1.0 / np.sqrt(pi)) * np.sum(_HW * vals)

def mean_lognorm(mu: float, sigma: float) -> float:
    return float(np.exp(mu + 0.5 * sigma * sigma))

def var_lognorm(mu: float, sigma: float) -> float:
    m1 = np.exp(mu + 0.5 * sigma * sigma)
    return float((np.exp(sigma*sigma) - 1.0) * (m1**2))

def phi_weibull(s: complex, k: float, lam: float) -> complex:
    # phi(s) = ∫_0^∞ e^{-t} * exp[-s*lam*t^{1/k}] dt  (Gauss–Laguerre on t)
    g = np.exp(-s * lam * (_LX ** (1.0 / k)))
    return np.sum(_LW * g)

def mean_weibull(k: float, lam: float) -> float:
    return float(lam * mp_gamma(1.0 + 1.0 / k))

def var_weibull(k: float, lam: float) -> float:
    m1 = mean_weibull(k, lam)
    m2 = (lam**2) * float(mp_gamma(1.0 + 2.0 / k))
    return float(m2 - m1*m1)

def severity_phi(typ: str, s: complex) -> complex:
    kind, a, b = SEV[typ]
    if kind == "lognorm":
        return phi_lognorm(s, a, b)
    else:
        return phi_weibull(s, a, b)

def severity_mean(typ: str) -> float:
    kind, a, b = SEV[typ]
    if kind == "lognorm":
        return mean_lognorm(a, b)
    else:
        return mean_weibull(a, b)

def severity_var(typ: str) -> float:
    kind, a, b = SEV[typ]
    if kind == "lognorm":
        return var_lognorm(a, b)
    else:
        return var_weibull(a, b)

# -----------------------------
# Build L_S(s) from BE counts
# -----------------------------
def be_counts_and_exact(depth: int):
    pmf_counts, meta = joint_pmf_by_types(
        depth=depth,
        V=V,
        node_types=node_types,
        Q=DELTA_EDGES,
        direct_law=DIRECT_LAW,
        return_exact_sets=True,
        normalize=True,
    )
    type_order = meta["type_order"]             # list of type labels in col order of counts
    P_exact: Dict[FrozenSet[int], float] = meta["P_exact"]
    F0 = float(P_exact.get(frozenset(), 0.0))   # atom at zero loss
    return pmf_counts, type_order, P_exact, F0

def build_LS(pmf_counts: Dict[Tuple[int,...], float], type_order: List[str]) -> Callable[[complex], complex]:
    def LS(s: complex) -> complex:
        phi = [severity_phi(t, s) for t in type_order]
        total = 0.0 + 0.0j
        for counts, p in pmf_counts.items():
            prod = 1.0 + 0.0j
            for i, x in enumerate(counts):
                if x:
                    prod *= (phi[i] ** x)
            total += p * prod
        return total
    return LS

# -----------------------------
# E[S], Var[S] from BE counts (no inversion)
# -----------------------------
def counts_stats_from_pmf(pmf_counts: Dict[Tuple[int,...], float], type_order: List[str]):
    M = len(type_order)
    muX = np.zeros(M)
    for counts, p in pmf_counts.items():
        muX += p * np.array(counts, dtype=float)
    Cov = np.zeros((M, M))
    for counts, p in pmf_counts.items():
        x = np.array(counts, dtype=float)
        Cov += p * np.outer(x - muX, x - muX)
    return muX, Cov

def ES_and_SD_from_counts(pmf_counts, type_order) -> Tuple[float, float]:
    muX, Cov = counts_stats_from_pmf(pmf_counts, type_order)
    m1 = np.array([severity_mean(t) for t in type_order])
    v1 = np.array([severity_var(t) for t in type_order])
    ES  = float(np.dot(muX, m1))
    VarS = float(np.dot(muX, v1) + m1 @ Cov @ m1)  # total variance law
    SDS = VarS**0.5
    return ES, SDS

# -----------------------------
# de Hoog–style inversion (your routine)
# -----------------------------
def dehoog_inversion(F: Callable[[complex], complex], t: float, gamma: float = 1e-4, N: int = 64) -> float:
    if t <= 0.0:
        return 0.0
    h = pi / t
    s0 = gamma
    f0 = 0.5 * np.real(F(s0))
    terms = [np.real(F(s0 + 1j * k * h)) for k in range(1, N+1)]
    alt = [ ((-1)**k) * terms[k-1] for k in range(1, N+1) ]
    s = 0.0
    p = 1.0
    for a in alt:
        p *= 0.5
        s += p * a
    return float((np.exp(s0 * t) / t) * (f0 + s))

# ---------- Correct F and H inversions with F0 ----------
def invert_CDF_from_LS(LS: Callable[[complex], complex], y: float, F0: float, gamma: float=1e-4, N: int=64) -> float:
    # F(y) = F0 + L^{-1}{ LS(s) / s }(y)
    def G(s: complex) -> complex:
        return LS(s) / s
    val = dehoog_inversion(G, y, gamma=gamma, N=N)
    return float(min(1.0, max(0.0, F0 + val)))

def invert_H_from_LS(LS: Callable[[complex], complex], y: float, F0: float, gamma: float=1e-4, N: int=64) -> float:
    # L{H}(s) = (1 - F0 - LS(s)) / s^2
    def Hs(s: complex) -> complex:
        return (1.0 - F0 - LS(s)) / (s*s)
    val = dehoog_inversion(Hs, y, gamma=gamma, N=N)
    return float(max(0.0, val))

# VaR_0.99 by bisection using corrected CDF
def find_var_0p99(LS: Callable[[complex], complex], F0: float, y_min: float, y_max: float, gamma: float=1e-4, N: int=64) -> float:
    target = 0.99
    lo, hi = y_min, y_max
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        Fm = invert_CDF_from_LS(LS, mid, F0, gamma=gamma, N=N)
        if Fm >= target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

# GMD = 2 ∫ F(1-F) dy using corrected CDF
def _choose_ycap_for_gmd(LS, F0, var_0p99, gamma=1e-4, N=64):
    y_cap = max(10_000.0, 4.0 * var_0p99)
    for _ in range(8):
        Fcap = invert_CDF_from_LS(LS, y_cap, F0, gamma=gamma, N=N)
        if 1.0 - Fcap <= 1e-9:
            return y_cap
        y_cap *= 2.0
    return y_cap

def gmd_from_cdf(LS, F0, var_0p99, gamma=1e-4, N=64, rel_tol=1e-4):
    y_cap = _choose_ycap_for_gmd(LS, F0, var_0p99, gamma=gamma, N=N)
    n0 = 400
    ys = np.geomspace(1e-6, y_cap, num=n0)
    Fs = np.array([invert_CDF_from_LS(LS, float(y), F0, gamma=gamma, N=N) for y in ys])
    gmd_old = 2.0 * np.trapz(Fs * (1.0 - Fs), ys)
    for _ in range(3):
        ys_ref = np.geomspace(1e-6, y_cap, num=len(ys)*2 - 1)
        Fs_ref = np.array([invert_CDF_from_LS(LS, float(y), F0, gamma=gamma, N=N) for y in ys_ref])
        gmd_new = 2.0 * np.trapz(Fs_ref * (1.0 - Fs_ref), ys_ref)
        if abs(gmd_new - gmd_old) <= rel_tol * max(1.0, gmd_new):
            return float(gmd_new)
        ys, gmd_old = ys_ref, gmd_new
    return float(gmd_old)

# -----------------------------
# Pricing for L in {3,4,5}
# -----------------------------
def pricing_row_for_L(L: int, theta: float=0.10, lam: float=0.15):
    # BE counts & LST
    pmf_counts, type_order, P_exact, F0 = be_counts_and_exact(L)
    LS = build_LS(pmf_counts, type_order)

    # Mean & SD (no inversion)
    ES, SDS = ES_and_SD_from_counts(pmf_counts, type_order)

    # VaR_0.99 & TVaR_0.99 with F0-aware inversions
    y_var = find_var_0p99(LS, F0, 1e4, 2e6, gamma=1e-4, N=64)
    Hy = invert_H_from_LS(LS, y_var, F0, gamma=1e-4, N=64)
    TVaR = y_var + (ES - Hy) / 0.01

    # GMD from CDF (F0-aware)
    GMD = gmd_from_cdf(LS, F0, y_var, gamma=1e-4, N=64)

    # Premium principles
    rho1 = (1.0 + theta) * ES
    rho2 = ES + theta * SDS
    rho3 = ES + theta * GMD
    pi_TVaR = (1.0 + lam) * TVaR

    return {
        "L": L,
        "E[S]": ES,
        "SD(S)": SDS,
        "GMD(S)": GMD,
        "VaR_0.99": y_var,
        "TVaR_0.99": TVaR,
        "rho1_exp": rho1,
        "rho2_sd": rho2,
        "rho3_gmd": rho3,
        "pi_TVaR": pi_TVaR,
        "F0": F0,
    }

def main():
    rows = [pricing_row_for_L(L) for L in (3,4,5)]
    df = pd.DataFrame(rows)
    # nice formatting
    def fmt0(x): return f"{x:,.0f}"
    out = pd.DataFrame({
        "L": df["L"],
        "F0": df["F0"].map(lambda x: f"{x:.6f}"),
        "E[S] (USD)": df["E[S]"].map(fmt0),
        "sqrt(Var) (USD)": df["SD(S)"].map(fmt0),
        "GMD (USD)": df["GMD(S)"].map(fmt0),
        "VaR_0.99 (USD)": df["VaR_0.99"].map(fmt0),
        "TVaR_0.99 (USD)": df["TVaR_0.99"].map(fmt0),
        "rho1 (Exp.)": df["rho1_exp"].map(fmt0),
        "rho2 (SD)": df["rho2_sd"].map(fmt0),
        "rho3 (GMD)": df["rho3_gmd"].map(fmt0),
        "pi_TVaR (TVaR-loaded)": df["pi_TVaR"].map(fmt0),
    })
    print("\nPremium sensitivity to containment depth L (BE–Laplace):")
    print(out.to_string(index=False))
    out.to_csv("pricing_vs_L.csv", index=False)
    print("\nSaved: pricing_vs_L.csv")

if __name__ == "__main__":
    main()
