#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TVaR_0.99 computation for the IIoT–SCADA case study using BE exact-set outputs,
with severities exactly as specified.

- CI/CD, IoT, FUS, ERP, SCADA ~ Lognormal(mu, sigma)  with mu = ln(median)
- SAMM, AD                     ~ Weibull(k, lambda)

Lognormal LST via Gauss–Hermite over Z ~ N(0,1)
Weibull   LST via Gauss–Laguerre on t in (0, inf)

This version:
- fixes the lognormal LST quadrature,
- handles F0 = P{S=0} correctly in inverse Laplace for CDF and H,
- uses proportional (multiplicative) reductions for edge controls.
"""

from __future__ import annotations
import numpy as np
from math import pi
from typing import Dict, Tuple, FrozenSet, Callable, List
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from mpmath import invertlaplace, gamma as mp_gamma  # robust de Hoog & Weibull mean

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

    # Weibull(k, lambda) with pdf f(y)=(k/λ)(y/λ)^{k-1} e^{-(y/λ)^k}
    "SAMM":  {"dist": "weibull", "k": 0.8, "lam":  90_000.0},
    "AD":    {"dist": "weibull", "k": 0.9, "lam": 110_000.0},
}

# -----------------------------
# 2) LSTs via quadrature
# -----------------------------
_HERM_N = 64
_HX, _HW = hermgauss(_HERM_N)  # nodes/weights for weight e^{-x^2} on (-inf, inf)

def phi_lognorm(s: complex, mu: float, sigma: float) -> complex:
    """
    LST(s) = E[e^{-s * exp(mu + sigma Z)}], Z~N(0,1)
    Gauss–Hermite: E[f(Z)] = (1/sqrt(pi)) * sum w_i * f(sqrt(2)*x_i)
    """
    z = np.sqrt(2.0) * _HX
    vals = np.exp(-s * np.exp(mu + sigma * z))
    return (1.0 / np.sqrt(pi)) * np.sum(_HW * vals)

_LAG_N = 64
_LX, _LW = laggauss(_LAG_N)  # nodes/weights for weight e^{-t} on (0, inf)

def phi_weibull(s: complex, k: float, lam: float) -> complex:
    """
    With t = (y/lam)^k  => LST = ∫_0^∞ e^{-t} * exp[-s * lam * t^{1/k}] dt
    Use Gauss–Laguerre on weight e^{-t}.
    """
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
    P_exact = meta["P_exact"]  # dict: frozenset(nodes) -> probability
    F0 = float(P_exact.get(frozenset(), 0.0))  # atom at zero-loss scenario
    return pmf, type_order, type_index, P_exact, F0

# -----------------------------
# 4) Build L_S(s) via counts PMF
# -----------------------------
def compound_LS_builder(pmf_counts: Dict[Tuple[int,...], float], type_order: List[str]) -> Callable[[complex], complex]:
    """
    L_S(s) = sum_{x} P{X=x} * prod_i phi_i(s)^{x_i}
    """
    def LS(s: complex) -> complex:
        phi = [severity_phi(t, s) for t in type_order]
        total = 0.0 + 0.0j
        for counts, prob in pmf_counts.items():
            prod_phi = 1.0 + 0.0j
            for i, x in enumerate(counts):
                if x:
                    prod_phi *= (phi[i] ** x)
            total += prob * prod_phi
        return total
    return LS

# -----------------------------
# 5) Inversion helpers (use mpmath's de Hoog)
# -----------------------------
def invert_CDF_from_LS(LS: Callable[[complex], complex], y: float, F0: float) -> float:
    """
    F(y) = F0 + L^{-1}{ L_S(s) / s }(y)
    """
    g = lambda s: LS(s) / s
    val = float(invertlaplace(g, y, method='dehoog'))
    Fy = F0 + val
    # numeric guard
    return float(min(1.0, max(0.0, Fy)))

def invert_H_from_LS(LS: Callable[[complex], complex], y: float, F0: float) -> float:
    """
    L{H}(s) = (1 - F0 - L_S(s)) / s^2
    """
    g = lambda s: (1.0 - F0 - LS(s)) / (s * s)
    Hy = float(invertlaplace(g, y, method='dehoog'))
    return float(max(0.0, Hy))

# -----------------------------
# 6) VaR and TVaR
# -----------------------------
def find_var(LF: Callable[[float], float], alpha: float, lo: float = 1e3, hi: float = 2e7, iters: int = 70) -> float:
    # expand hi if necessary
    while LF(hi) < alpha and hi < 1e12:
        hi *= 2.0
    a, b = lo, hi
    for _ in range(iters):
        m = 0.5 * (a + b)
        if LF(m) >= alpha:
            b = m
        else:
            a = m
    return 0.5 * (a + b)

def compute_mean_S(P_exact: Dict[FrozenSet[int], float], node_types_map: Dict[int, str]) -> float:
    """
    E[S] = sum_i E[X_i] * E[Y_i].
    E[X_i] from exact sets by counting nodes of type i in each exact set.
    """
    types = sorted(set(node_types_map.values()))
    idx = {t:i for i,t in enumerate(types)}
    ex = np.zeros(len(types), dtype=float)
    for C, p in P_exact.items():
        for n in C:
            ex[idx[node_types_map[n]]] += p
    ES = 0.0
    for t, i in idx.items():
        ES += ex[i] * severity_mean(t)
    return ES

def compute_var_tvar(
    depth: int,
    Q: Dict[Tuple[int,int], float],
    direct_law: Dict[FrozenSet[int], float],
    node_types_map: Dict[int,str],
    alpha: float = 0.99,
    var_bracket: Tuple[float, float] = (1e4, 2e7),
):
    pmf_counts, type_order, type_index, P_exact, F0 = exact_counts_pmf(depth, Q, direct_law, node_types_map)
    LS = compound_LS_builder(pmf_counts, type_order)
    ES = compute_mean_S(P_exact, node_types_map)

    F = lambda y: invert_CDF_from_LS(LS, y, F0)
    v_alpha = find_var(F, alpha, lo=var_bracket[0], hi=var_bracket[1], iters=70)
    H_at_v = invert_H_from_LS(LS, v_alpha, F0)
    TVaR = v_alpha + (ES - H_at_v) / (1.0 - alpha)
    return ES, v_alpha, H_at_v, TVaR

# -----------------------------
# 7) Controls: proportional reductions + segmentation
# -----------------------------
def scale_edge_proportional(Q: Dict[Tuple[int,int], float], edge: Tuple[int,int], pct_reduction: float) -> Dict[Tuple[int,int], float]:
    """
    Multiply the edge probability by (1 - pct_reduction).
    pct_reduction = 0.10 -> new p = 0.90 * old p
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
    ES, VaR, HVaR, TVaR = compute_var_tvar(
        depth=BASELINE_L,
        Q=DELTA_EDGES_BASE,
        direct_law=DIRECT_LAW,
        node_types_map=node_types,
        alpha=0.99,
        var_bracket=(1e4, 2e7),
    )
    print(f"Baseline L={BASELINE_L}: E[S]={ES:,.0f}, VaR_0.99={VaR:,.0f}, H(VaR)={HVaR:,.0f}, TVaR_0.99={TVaR:,.0f}")

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
        ES2, VaR2, HVaR2, TVaR2 = compute_var_tvar(
            depth=L2, Q=Q2, direct_law=DIRECT_LAW, node_types_map=node_types,
            alpha=0.99, var_bracket=(1e4, 2e7)
        )
        rows.append({
            "Control": c["name"],
            "ΔTVaR": TVaR - TVaR2,
        })

    df = pd.DataFrame(rows)
    df_print = df.copy()
    df_print["ΔTVaR"] = df_print["ΔTVaR"].map(lambda x: f"{x:,.2f}")

    print("\nTail–risk impact of individual controls (proportional reductions):")
    print(df_print.to_string(index=False))

if __name__ == "__main__":
    main()
