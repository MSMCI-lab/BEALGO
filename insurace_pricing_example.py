#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Insurance pricing table from BE-derived loss distribution (L in {3,4,5}).

- Imports BE algorithms from be_joint_pmf.py
- Uses user-specified severities:
    CI/CD, IoT, FUS, ERP, SCADA ~ Lognormal(mu, sigma)
    SAMM, AD                    ~ Weibull(k, lambda)
- Computes E[Y], SD(Y), GMD, VaR_0.99, TVaR_0.99 via Laplace inversion,
  then premium principles (Expectation, SD, GMD, TVaR-loaded).

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
    "SAMM":   ("weibull", 0.8, 90_000.0),   # (k, lambda)
    "AD":     ("weibull", 0.9, 110_000.0),
}

# Quadrature nodes for LSTs
_HERM_N = 64
_HX, _HW = hermgauss(_HERM_N)      # for lognormal (Gauss–Hermite)
_LAG_N = 64
_LX, _LW = laggauss(_LAG_N)        # for Weibull   (Gauss–Laguerre)

def phi_lognorm(s: complex, mu: float, sigma: float) -> complex:
    # Y = exp(Z), Z~N(mu, sigma^2): phi(s)=E[e^{-s e^Z}]
    a = sigma * np.sqrt(2.0)
    z = mu + a * _HX
    vals = np.exp(-s * np.exp(z))
    return (a / np.sqrt(pi)) * np.sum(_HW * vals)

def mean_lognorm(mu: float, sigma: float) -> float:
    return float(np.exp(mu + 0.5 * sigma * sigma))

def var_lognorm(mu: float, sigma: float) -> float:
    m1 = np.exp(mu + 0.5 * sigma * sigma)
    return float((np.exp(sigma*sigma) - 1.0) * (m1**2))

def phi_weibull(s: complex, k: float, lam: float) -> complex:
    # phi(s) = ∫_0^∞ e^{-t - s*lam*t^{1/k}} dt  (Gauss–Laguerre)
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
# Build L_Y(s) from BE counts
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
    type_order = meta["type_order"]             # list of type labels in column order of counts
    P_exact: Dict[FrozenSet[int], float] = meta["P_exact"]
    return pmf_counts, type_order, P_exact

def build_LY(pmf_counts: Dict[Tuple[int,...], float], type_order: List[str]) -> Callable[[complex], complex]:
    def LY(s: complex) -> complex:
        phi = [severity_phi(t, s) for t in type_order]
        total = 0.0 + 0.0j
        for counts, p in pmf_counts.items():
            prod = 1.0 + 0.0j
            for i, x in enumerate(counts):
                if x:
                    prod *= (phi[i] ** x)
            total += p * prod
        return total
    return LY

# -----------------------------
# E[Y], Var[Y] from BE counts pmf (no inversion)
# -----------------------------
def counts_stats_from_pmf(pmf_counts: Dict[Tuple[int,...], float], type_order: List[str]):
    # mean vector
    M = len(type_order)
    muX = np.zeros(M)
    for counts, p in pmf_counts.items():
        muX += p * np.array(counts, dtype=float)
    # covariance
    Cov = np.zeros((M, M))
    for counts, p in pmf_counts.items():
        x = np.array(counts, dtype=float)
        Cov += p * np.outer(x - muX, x - muX)
    return muX, Cov

def EY_and_SD_from_counts(pmf_counts, type_order) -> Tuple[float, float]:
    muX, Cov = counts_stats_from_pmf(pmf_counts, type_order)
    m1 = np.array([severity_mean(t) for t in type_order])
    v1 = np.array([severity_var(t) for t in type_order])
    # Law of total variance:
    # Var(Y) = E[Var(Y|X)] + Var(E[Y|X]) = sum_i E[X_i]*Var_i + m^T Cov(X) m
    EY = float(np.dot(muX, m1))
    VarY = float(np.dot(muX, v1) + m1 @ Cov @ m1)
    SDY = VarY**0.5
    return EY, SDY

# -----------------------------
# de Hoog–Knight–Stokes inversion
# -----------------------------
def dehoog_inversion(F: Callable[[complex], complex], t: float, gamma: float = 1e-4, N: int = 64) -> float:
    if t <= 0.0:
        return 0.0
    h = pi / t
    s0 = gamma
    f0 = 0.5 * np.real(F(s0))
    terms = []
    for k in range(1, N+1):
        sk = s0 + 1j * k * h
        terms.append(np.real(F(sk)))
    # alternating series + simple Euler acceleration
    alt = [ ((-1)**k) * terms[k-1] for k in range(1, N+1) ]
    s = 0.0
    p = 1.0
    for a in alt:
        p *= 0.5
        s += p * a
    return float((np.exp(s0 * t) / t) * (f0 + s))

def invert_CDF_from_LY(LY: Callable[[complex], complex], y: float, gamma: float=1e-4, N: int=64) -> float:
    def LF(s: complex) -> complex:
        return (1.0 - LY(s)) / s
    F = dehoog_inversion(LF, y, gamma=gamma, N=N)
    return max(0.0, min(1.0, F))

def invert_H_from_LY(LY: Callable[[complex], complex], y: float, gamma: float=1e-4, N: int=64) -> float:
    def LH(s: complex) -> complex:
        return (1.0 - LY(s)) / (s*s)
    H = dehoog_inversion(LH, y, gamma=gamma, N=N)
    return max(0.0, H)

# VaR_0.99 by bisection
def find_var_0p99(LY: Callable[[complex], complex], y_min: float, y_max: float, gamma: float=1e-4, N: int=64) -> float:
    target = 0.99
    lo, hi = y_min, y_max
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        Fm = invert_CDF_from_LY(LY, mid, gamma=gamma, N=N)
        if Fm >= target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

# GMD = 2 ∫ F(1-F) dy via adaptive geometric grid
def _choose_ycap_for_gmd(LY, VaR_0p99, gamma=1e-4, N=64):
    y_cap = max(10_000.0, 4.0 * VaR_0p99)
    for _ in range(8):
        Fcap = invert_CDF_from_LY(LY, y_cap, gamma=gamma, N=N)
        if 1.0 - Fcap <= 1e-9:
            return y_cap
        y_cap *= 2.0
    return y_cap

def gmd_from_cdf(LY, VaR_0p99, gamma=1e-4, N=64, rel_tol=1e-4):
    y_cap = _choose_ycap_for_gmd(LY, VaR_0p99, gamma=gamma, N=N)
    n0 = 400
    ys = np.geomspace(1e-6, y_cap, num=n0)
    Fs = np.array([invert_CDF_from_LY(LY, float(y), gamma=gamma, N=N) for y in ys])
    integrand = Fs * (1.0 - Fs)
    gmd_old = 2.0 * np.trapz(integrand, ys)
    for _ in range(3):
        ys_ref = np.geomspace(1e-6, y_cap, num=len(ys)*2 - 1)
        Fs_ref = np.array([invert_CDF_from_LY(LY, float(y), gamma=gamma, N=N) for y in ys_ref])
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
    pmf_counts, type_order, P_exact = be_counts_and_exact(L)
    LY = build_LY(pmf_counts, type_order)

    # Mean & SD (no inversion)
    EY, SDY = EY_and_SD_from_counts(pmf_counts, type_order)

    # VaR_0.99 & TVaR_0.99
    y_var = find_var_0p99(LY, 1e4, 2e6, gamma=1e-4, N=64)
    Hy = invert_H_from_LY(LY, y_var, gamma=1e-4, N=64)
    TVaR = y_var + (EY - Hy) / 0.01

    # GMD from CDF
    GMD = gmd_from_cdf(LY, y_var, gamma=1e-4, N=64)

    # Premium principles
    rho1 = (1.0 + theta) * EY
    rho2 = EY + theta * SDY
    rho3 = EY + theta * GMD
    pi_TVaR = (1.0 + lam) * TVaR

    return {
        "L": L,
        "E[Y]": EY,
        "SD(Y)": SDY,
        "GMD(Y)": GMD,
        "VaR_0.99": y_var,
        "TVaR_0.99": TVaR,
        "rho1_exp": rho1,
        "rho2_sd": rho2,
        "rho3_gmd": rho3,
        "pi_TVaR": pi_TVaR,
    }

def main():
    rows = [pricing_row_for_L(L) for L in (3,4,5)]
    df = pd.DataFrame(rows)
    # nice formatting
    def fmt(x): return f"{x:,.0f}"
    out = pd.DataFrame({
        "L": df["L"],
        "E[Y] (USD)": df["E[Y]"].map(fmt),
        "sqrt(Var) (USD)": df["SD(Y)"].map(fmt),
        "GMD (USD)": df["GMD(Y)"].map(fmt),
        "VaR_0.99 (USD)": df["VaR_0.99"].map(fmt),
        "TVaR_0.99 (USD)": df["TVaR_0.99"].map(fmt),
        "rho1 (Exp.)": df["rho1_exp"].map(fmt),
        "rho2 (SD)": df["rho2_sd"].map(fmt),
        "rho3 (GMD)": df["rho3_gmd"].map(fmt),
        "pi_TVaR (TVaR-loaded)": df["pi_TVaR"].map(fmt),
    })
    print("\nPremium sensitivity to containment depth L (BE–Laplace):")
    print(out.to_string(index=False))
    out.to_csv("pricing_vs_L.csv", index=False)
    print("\nSaved: pricing_vs_L.csv")

if __name__ == "__main__":
    main()
