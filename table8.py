#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TVaR_0.99 computation for the IIoT–SCADA case study using BE exact-set outputs,
with severities exactly as specified by the user.

- CI/CD, IoT, FUS, ERP, SCADA ~ Lognormal(mu, sigma)
- SAMM, AD                     ~ Weibull(k, lambda)

Lognormal LST phi(s) via Gauss–Hermite over Z ~ N(mu, sigma^2)
Weibull   LST phi(s) via Gauss–Laguerre on t in (0, inf)

This version uses PROPORTIONAL (multiplicative) reductions for edge controls
(e.g., 10% reduction means new p = 0.9 * old p), and prints a compact
two-column table: Control, ΔTVaR.
"""

from __future__ import annotations
import numpy as np
from math import pi
from typing import Dict, Tuple, FrozenSet, Callable, List
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from mpmath import gamma as mp_gamma  # Weibull mean

# --- import BE module (Algorithms 1–2) ---
from be_joint_pmf import joint_pmf_by_types

# -----------------------------
# 0) Network and BE inputs
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

# Corrected Table 6: ONLY these 8 edges exist
DELTA_EDGES_BASE: Dict[Tuple[int,int], float] = {
    (1,2): 0.10,
    (2,3): 0.10,
    (3,4): 0.30,
    (4,7): 0.10,
    (5,6): 0.20,
    (6,7): 0.15,
    (1,4): 0.15,
    (4,6): 0.60,
}

# Direct-entry law (entries at nodes 1 and 5)
alpha1, alpha5 = 0.12, 0.08
DIRECT_LAW = {
    frozenset(): (1 - alpha1) * (1 - alpha5),
    frozenset({1}): alpha1 * (1 - alpha5),
    frozenset({5}): (1 - alpha1) * alpha5,
    frozenset({1, 5}): alpha1 * alpha5,
}

BASELINE_L = 5

# -----------------------------
# 1) Severities (exactly as provided)
# -----------------------------
TYPE_SPEC = {
    # Lognormal(mu, sigma) where mu = ln(median)
    "CI/CD": {"dist": "lognorm", "mu": np.log(30_000.0), "sigma": 0.6},
    "IoT":   {"dist": "lognorm", "mu": np.log(25_000.0), "sigma": 0.6},
    "FUS":   {"dist": "lognorm", "mu": np.log(40_000.0), "sigma": 0.7},
    "ERP":   {"dist": "lognorm", "mu": np.log(80_000.0), "sigma": 0.9},
    "SCADA": {"dist": "lognorm", "mu": np.log(150_000.0), "sigma": 1.0},

    # Weibull(k, lambda)
    "SAMM":  {"dist": "weibull", "k": 0.8, "lam": 90_000.0},
    "AD":    {"dist": "weibull", "k": 0.9, "lam":110_000.0},
}

# -----------------------------
# 2) LSTs via quadrature
# -----------------------------
_HERM_N = 64
_HX, _HW = hermgauss(_HERM_N)

def phi_lognorm(s: complex, mu: float, sigma: float) -> complex:
    a = sigma * np.sqrt(2.0)
    z = mu + a * _HX
    vals = np.exp(-s * np.exp(z))
    return (a / np.sqrt(pi)) * np.sum(_HW * vals)

_LAG_N = 64
_LX, _LW = laggauss(_LAG_N)

def phi_weibull(s: complex, k: float, lam: float) -> complex:
    g = np.exp(-s * lam * (_LX ** (1.0 / k)))
    return np.sum(_LW * g)

# Means for E[Y]
def mean_lognorm(mu: float, sigma: float) -> float:
    return float(np.exp(mu + 0.5 * sigma * sigma))

def mean_weibull(k: float, lam: float) -> float:
    return float(lam * mp_gamma(1.0 + 1.0 / k))

def severity_phi(typ: str, s: complex) -> complex:
    spec = TYPE_SPEC[typ]
    if spec["dist"] == "lognorm":
        return phi_lognorm(s, spec["mu"], spec["sigma"])
    elif spec["dist"] == "weibull":
        return phi_weibull(s, spec["k"], spec["lam"])
    raise ValueError(f"Unknown dist for type {typ}")

def severity_mean(typ: str) -> float:
    spec = TYPE_SPEC[typ]
    if spec["dist"] == "lognorm":
        return mean_lognorm(spec["mu"], spec["sigma"])
    elif spec["dist"] == "weibull":
        return mean_weibull(spec["k"], spec["lam"])
    raise ValueError(f"Unknown dist for type {typ}")

# -----------------------------
# 3) BE exact-set aggregation
# -----------------------------
def exact_counts_pmf(
    depth: int,
    Q: Dict[Tuple[int,int], float],
    direct_law: Dict[FrozenSet[int], float],
    node_types_map: Dict[int, str],
):
    pmf, meta = joint_pmf_by_types(
        depth=depth,
        V=V,
        node_types=node_types_map,
        Q=Q,
        direct_law=direct_law,
        return_exact_sets=True,
        normalize=True,
    )
    type_order = meta["type_order"]
    type_index = {t: i for i, t in enumerate(type_order)}
    P_exact = meta["P_exact"]
    return pmf, type_order, type_index, P_exact

# -----------------------------
# 4) Build L_Y(s) via counts PMF
# -----------------------------
def compound_LY_builder(pmf_counts: Dict[Tuple[int,...], float], type_order: List[str]) -> Callable[[complex], complex]:
    def LY(s: complex) -> complex:
        phi = [severity_phi(t, s) for t in type_order]
        total = 0.0 + 0.0j
        for counts, prob in pmf_counts.items():
            prod_phi = 1.0 + 0.0j
            for i, x in enumerate(counts):
                if x:
                    prod_phi *= (phi[i] ** x)
            total += prob * prod_phi
        return total
    return LY

# -----------------------------
# 5) de Hoog–Knight–Stokes inversion
# -----------------------------
def dehoog_inversion(F: Callable[[complex], complex], t: float, gamma: float = 1e-4, N: int = 64) -> float:
    if t <= 0.0:
        return 0.0
    h = pi / t
    s0 = gamma
    f0 = 0.5 * np.real(F(s0))
    terms = [np.real(F(s0 + 1j * k * h)) for k in range(1, N+1)]
    alt = [((-1)**k) * terms[k-1] for k in range(1, N+1)]
    s = 0.0
    p = 1.0
    for a in alt:
        p *= 0.5
        s += p * a
    return float((np.exp(s0 * t) / t) * (f0 + s))

def invert_CDF_from_LY(LY: Callable[[complex], complex], y: float, gamma: float = 1e-4, N: int = 64) -> float:
    def LF(s: complex) -> complex:
        return (1.0 - LY(s)) / s
    return max(0.0, min(1.0, dehoog_inversion(LF, y, gamma=gamma, N=N)))

def invert_H_from_LY(LY: Callable[[complex], complex], y: float, gamma: float = 1e-4, N: int = 64) -> float:
    def LH(s: complex) -> complex:
        return (1.0 - LY(s)) / (s * s)
    return max(0.0, dehoog_inversion(LH, y, gamma=gamma, N=N))

# -----------------------------
# 6) VaR and TVaR
# -----------------------------
def find_var_0p99(LY: Callable[[complex], complex], y_min: float, y_max: float, gamma: float = 1e-4, N: int = 64) -> float:
    target = 0.99
    lo, hi = y_min, y_max
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        Fmid = invert_CDF_from_LY(LY, mid, gamma=gamma, N=N)
        if Fmid >= target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def compute_mean_Y(P_exact: Dict[FrozenSet[int], float], node_types_map: Dict[int, str]) -> float:
    types = sorted(set(node_types_map.values()))
    idx = {t:i for i,t in enumerate(types)}
    ex = np.zeros(len(types), dtype=float)
    for C, p in P_exact.items():
        for n in C:
            ex[idx[node_types_map[n]]] += p
    EY = 0.0
    for t, i in idx.items():
        EY += ex[i] * severity_mean(t)
    return EY

def compute_var_tvar(
    depth: int,
    Q: Dict[Tuple[int,int], float],
    direct_law: Dict[FrozenSet[int], float],
    node_types_map: Dict[int,str],
    gamma: float = 1e-4,
    N: int = 64,
    var_bracket: Tuple[float, float] = (1e4, 2e6),
):
    pmf_counts, type_order, type_index, P_exact = exact_counts_pmf(depth, Q, direct_law, node_types_map)
    LY = compound_LY_builder(pmf_counts, type_order)
    EY = compute_mean_Y(P_exact, node_types_map)
    y_var = find_var_0p99(LY, var_bracket[0], var_bracket[1], gamma=gamma, N=N)
    Hy = invert_H_from_LY(LY, y_var, gamma=gamma, N=N)
    TVaR = y_var + (EY - Hy) / (1.0 - 0.99)
    return EY, y_var, Hy, TVaR

# -----------------------------
# 7) Controls: proportional reductions + segmentation
# -----------------------------
def scale_edge_proportional(Q: Dict[Tuple[int,int], float], edge: Tuple[int,int], pct_reduction: float) -> Dict[Tuple[int,int], float]:
    """
    Multiply the edge probability by (1 - pct_reduction).
    pct_reduction = 0.10 -> 10% reduction -> new p = 0.90 * old p
    """
    Q2 = dict(Q)
    if edge in Q2:
        factor = max(0.0, min(1.0, 1.0 - float(pct_reduction)))
        Q2[edge] = max(0.0, min(1.0, Q2[edge] * factor))
    return Q2

def reduce_depth(L: int, hops: int) -> int:
    return max(0, L - hops)

def main():
    print("=== TVaR with specified severities (Lognormal & Weibull) ===")
    EY, VaR, HVaR, TVaR = compute_var_tvar(
        depth=BASELINE_L,
        Q=DELTA_EDGES_BASE,
        direct_law=DIRECT_LAW,
        node_types_map=node_types,
        gamma=1e-4, N=64,
        var_bracket=(1e4, 2e6)
    )
    print(f"E[Y]={EY:,.0f}, VaR_0.99={VaR:,.0f}, H(VaR)={HVaR:,.0f}, TVaR_0.99={TVaR:,.0f}")

    # Proportional controls (edit percentages as desired)
    controls = [
        {"name": "Rate limiting: IoT ingress (1→2) −20%", "kind": "edge",  "edge": (1,2), "pct": 0.20},
        {"name": "Edge hardening: SAMM→SCADA (4→7) −20%", "kind": "edge",  "edge": (4,7), "pct": 0.20},
        {"name": "Patch cluster: AD DCs (6→7) −20%",       "kind": "edge",  "edge": (6,7), "pct": 0.20},
        {"name": "Edge hardening: IoT→SAMM (3→4) −20%",    "kind": "edge",  "edge": (3,4), "pct": 0.20},
        {"name": "Segmentation: L=5→4",                    "kind": "depth", "hops": 1},
    ]

    rows = []
    for c in controls:
        if c["kind"] == "edge":
            Q2 = scale_edge_proportional(DELTA_EDGES_BASE, c["edge"], c["pct"])
            L2 = BASELINE_L
        else:
            Q2 = DELTA_EDGES_BASE
            L2 = reduce_depth(BASELINE_L, c["hops"])
        EY2, VaR2, HVaR2, TVaR2 = compute_var_tvar(
            depth=L2, Q=Q2, direct_law=DIRECT_LAW, node_types_map=node_types,
            gamma=1e-4, N=64, var_bracket=(1e4, 2e6)
        )
        rows.append({
            "Control": c["name"],
            "ΔTVaR": TVaR - TVaR2,
        })

    df = pd.DataFrame(rows)
    df_print = df.copy()
    df_print["ΔTVaR"] = df_print["ΔTVaR"].map(lambda x: f"{x:,.6f}")

    print("\nTail–risk impact of individual controls (proportional reductions):")
    print(df_print.to_string(index=False))

if __name__ == "__main__":
    main()
