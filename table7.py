#!/usr/bin/env python3
# table7_scada_by_depth.py
from pathlib import Path
import pandas as pd
from be_joint_pmf import joint_pmf_by_types

OUT_CSV = Path("table7_scada_by_depth.csv")
SCADA_NODE = 7
DEPTHS = [1,2,3,4,5,6]

# direct entries as in Sec.6/6.1
alpha1, alpha5 = 0.12, 0.08
DIRECT_LAW = {
    frozenset(): (1-alpha1)*(1-alpha5),
    frozenset({1}): alpha1*(1-alpha5),
    frozenset({5}): (1-alpha1)*alpha5,
    frozenset({1,5}): alpha1*alpha5,
}

# corridors
CORRIDORS = {
    "P1": [(1,2),(2,3),(3,4),(4,7)],
    "P2": [(5,6),(6,7)],
    "P3": [(1,4),(4,6),(6,7)],
    "P4": [(1,4),(4,7)],
}
C_UNION_NODES = sorted({n for edges in CORRIDORS.values() for e in edges for n in e})

# corrected Table 6 — only these eight edges exist
DELTA_EDGES = {
    (1,2): 0.10,
    (2,3): 0.10,
    (3,4): 0.30,
    (4,7): 0.10,
    (5,6): 0.20,
    (6,7): 0.15,
    (1,4): 0.15,
    (4,6): 0.60,
}

def scale_Q(Q, factor: float):
    return {e: max(0.0, min(1.0, q*factor)) for e,q in Q.items()}

def subgraph(nodes, Q):
    S = set(nodes)
    return {e: q for e,q in Q.items() if (e[0] in S and e[1] in S)}

def exact_prob_scada_by_depth(V, Q, direct_law, L, scada_node=SCADA_NODE):
    node_types = {n: n for n in V}
    _, meta = joint_pmf_by_types(
        depth=L, V=V, node_types=node_types, Q=Q,
        direct_law=direct_law, return_exact_sets=True, normalize=True
    )
    P_exact = meta["P_exact"]
    return sum(p for C,p in P_exact.items() if scada_node in C)

def corridor_prob_by_depth(corr_edges, Q_all, direct_law, L):
    nodes = sorted({n for e in corr_edges for n in e})
    Qc = {e: Q_all[e] for e in corr_edges if e in Q_all}
    # restrict direct law to corridor nodes
    dl = {}
    S = set(nodes)
    for D0, pr in direct_law.items():
        D0_corr = frozenset(set(D0) & S)
        dl[D0_corr] = dl.get(D0_corr, 0.0) + pr
    return exact_prob_scada_by_depth(nodes, Qc, dl, L)

def main():
    Q_union = subgraph(C_UNION_NODES, DELTA_EDGES)
    Q_corr = DELTA_EDGES  # corridor skeleton

    rows = []
    for L in DEPTHS:
        p_corr = {name: corridor_prob_by_depth(edges, Q_corr, DIRECT_LAW, L)
                  for name, edges in CORRIDORS.items()}
        pmax_corr = max(p_corr.values()) if p_corr else 0.0

        p_scada = exact_prob_scada_by_depth(C_UNION_NODES, Q_union, DIRECT_LAW, L)

        # ±20%
        Q_corr_lo = scale_Q(Q_corr, 0.8); Q_corr_hi = scale_Q(Q_corr, 1.2)
        Q_union_lo = scale_Q(Q_union, 0.8); Q_union_hi = scale_Q(Q_union, 1.2)

        pmax_lo = max(corridor_prob_by_depth(edges, Q_corr_lo, DIRECT_LAW, L) for edges in CORRIDORS.values())
        pmax_hi = max(corridor_prob_by_depth(edges, Q_corr_hi, DIRECT_LAW, L) for edges in CORRIDORS.values())

        p_scada_lo = exact_prob_scada_by_depth(C_UNION_NODES, Q_union_lo, DIRECT_LAW, L)
        p_scada_hi = exact_prob_scada_by_depth(C_UNION_NODES, Q_union_hi, DIRECT_LAW, L)

        pct_delta = ("{:.1f}%".format(100.0*(p_scada_hi - p_scada_lo)/p_scada) if p_scada > 0 else "—")

        rows.append({
            "L": L,
            "Pmax corridor (baseline)": round(pmax_corr, 6),
            "PSCADA (baseline)": round(p_scada, 6),
            "Pmax corridor [L,U]": f"[{pmax_lo:.6f}, {pmax_hi:.6f}]",
            "PSCADA [L,U]": f"[{p_scada_lo:.6f}, {p_scada_hi:.6f}]",
            "% Δ": pct_delta,
        })

    df = pd.DataFrame(rows, columns=[
        "L","Pmax corridor (baseline)","PSCADA (baseline)",
        "Pmax corridor [L,U]","PSCADA [L,U]","% Δ"
    ])
    print("\nTable 7 — SCADA compromise probabilities by containment depth")
    print(df.to_string(index=False))
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()
