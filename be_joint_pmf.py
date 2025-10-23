# be_joint_pmf.py
# ------------------------------------------------------------
# Exact joint PMF via Backward Elimination (BE) for L-hop model
# ------------------------------------------------------------
# Public API:
#   - pi_from_independent_p(V, p)
#   - joint_pmf_by_types(depth, V, node_types, Q, direct_law=None, independent_p=None,
#                        return_exact_sets=False, normalize=True, candidate_sets=None)
#
# Example:
#   from be_joint_pmf import joint_pmf_by_types, pi_from_independent_p
#   pmf, meta = joint_pmf_by_types(L, V, node_types, Q, independent_p=p)
# ------------------------------------------------------------

from itertools import combinations
from typing import Dict, Iterable, Tuple, Set, FrozenSet, Optional, Any, Callable

__all__ = [
    "pi_from_independent_p",
    "joint_pmf_by_types",
]

# --------------------------
# Utilities
# --------------------------

def _powerset(s: Iterable[Any]):
    """All subsets of s as frozensets."""
    s = tuple(s)
    for r in range(len(s) + 1):
        for comb in combinations(s, r):
            yield frozenset(comb)

def _product(iterable):
    out = 1.0
    for v in iterable:
        out *= v
    return out

# --------------------------
# Direct-entry law π(D0)
# --------------------------

def pi_from_independent_p(
    V: Iterable[Any],
    p: Dict[Any, float],
) -> Dict[FrozenSet[Any], float]:
    """
    Build π(D0) assuming node-wise independent direct attacks with probs p[i].
    π(D0) = ∏_{i∈D0} p_i ∏_{i∉D0} (1-p_i)

    Returns
    -------
    dict {frozenset D0 -> probability}
    """
    V = tuple(V)
    pi = {}
    for D0 in _powerset(V):
        prob = 1.0
        for i in V:
            prob *= p[i] if i in D0 else (1.0 - p[i])
        pi[D0] = prob
    # light renormalization against FP drift
    tot = sum(pi.values())
    if tot > 0 and abs(tot - 1.0) > 1e-12:
        for k in pi:
            pi[k] /= tot
    return pi

# --------------------------
# BE kernels (Algorithm 1)
# --------------------------

class BEKernel:
    """
    Memoized BE kernels for γ^{(r)}_U(C | D0) on a directed graph with edge
    probabilities Q[(u,v)] = q_uv in [0,1]. Nodes are hashable.
    """

    def __init__(self, Q: Dict[Tuple[Any, Any], float]):
        self.Q = Q
        self._cache_g1 = {}  # (U,C,D) -> float
        self._cache_g  = {}  # (U,C,D0,r) -> float

    def _q(self, u, v) -> float:
        return float(self.Q.get((u, v), 0.0))

    def gamma_one(
        self,
        U: FrozenSet[Any],
        C: FrozenSet[Any],
        D: FrozenSet[Any],
    ) -> float:
        """
        One-round kernel γ^{(1)}_U(C | D):
          = [∏_{j∈(C\D)} (1 - ∏_{i∈D} (1 - q_{ij}))] * [∏_{u∈D} ∏_{v∈(U\C)} (1 - q_{uv})]

        Edge cases:
          - If D is empty: γ^{(1)}=1 iff C==D, else 0 (no new infections can appear spontaneously).
        """
        key = (U, C, D)
        if key in self._cache_g1:
            return self._cache_g1[key]

        if not D:
            val = 1.0 if C == D else 0.0
            self._cache_g1[key] = val
            return val

        new_targets = C - D
        # hit (all newly infected occur)
        if new_targets:
            phit = _product(
                1.0 - _product((1.0 - self._q(i, j)) for i in D)
                for j in new_targets
            )
        else:
            phit = 1.0

        # block (no spillover outside C)
        outside = U - C
        if outside:
            pblock = _product(1.0 - self._q(u, v) for u in D for v in outside)
        else:
            pblock = 1.0

        g = phit * pblock
        # small numeric guard
        if g < 0 and g > -1e-15:
            g = 0.0
        if g > 1.0:
            g = 1.0
        self._cache_g1[key] = g
        return g

    def gamma_multi(
        self,
        U: FrozenSet[Any],
        C: FrozenSet[Any],
        D0: FrozenSet[Any],
        r: int,
    ) -> float:
        """
        Recursive multi-round kernel γ^{(r)}_U(C | D0). For r >= 2:
          γ^{(r)}_U(C | D0) = sum_{D ⊆ (C\D0)} γ^{(1)}_U(D ∪ D0 | D0) * γ^{(r-1)}_{U\D0}(C\D0 | D)
        Base case (r==1): γ^{(1)}_U(C | D0).
        """
        if r < 1:
            raise ValueError("Depth r must be >= 1")
        key = (U, C, D0, r)
        if key in self._cache_g:
            return self._cache_g[key]

        if r == 1:
            val = self.gamma_one(U, C, D0)
            self._cache_g[key] = val
            return val

        if not D0.issubset(C):
            self._cache_g[key] = 0.0
            return 0.0

        s = 0.0
        Unext = U - D0
        Cnext = C - D0
        for D in _powerset(Cnext):
            g1 = self.gamma_one(U, D | D0, D0)
            if g1 == 0.0:
                continue
            g2 = self.gamma_multi(Unext, Cnext, D, r - 1)
            if g2 == 0.0:
                continue
            s += g1 * g2

        # numeric guard
        if s < 0 and s > -1e-15:
            s = 0.0
        self._cache_g[key] = s
        return s

# --------------------------
# Joint PMF (Algorithm 2)
# --------------------------

def joint_pmf_by_types(
    depth: int,
    V: Iterable[Any],
    node_types: Dict[Any, Any],
    Q: Dict[Tuple[Any, Any], float],
    direct_law: Optional[Dict[FrozenSet[Any], float]] = None,
    independent_p: Optional[Dict[Any, float]] = None,
    *,
    return_exact_sets: bool = False,
    normalize: bool = True,
    candidate_sets: Optional[Callable[[FrozenSet[Any], int], Iterable[FrozenSet[Any]]]] = None,
):
    """
    Compute the exact joint PMF f^{(depth)}(x) over type-count vectors x via BE (Algorithms 1 & 2).

    Parameters
    ----------
    depth : int
        Propagation depth L (number of rounds; L>=0). If L==0, returns distribution of direct sets only.
    V : iterable of nodes (hashable)
        Universe of nodes.
    node_types : dict {node -> type_label}
        Type labels (hashable). Determines aggregation.
    Q : dict {(u,v) -> q_uv in [0,1]}
        Edge-wise indirect compromise probabilities.
    direct_law : dict {frozenset D0 -> π(D0)}, optional
        Explicit direct-entry law. If omitted, 'independent_p' must be provided.
    independent_p : dict {node -> p_i}, optional
        Per-node independent direct probabilities; used if 'direct_law' is None.
    return_exact_sets : bool, default False
        If True, also return the exact-set probabilities {C -> Pr(B^{(L)}_C)}.
    normalize : bool, default True
        If True, lightly renormalize exact-set and PMF outputs to sum to 1 (guard FP drift).
    candidate_sets : callable (V_frozen, depth) -> iterable of frozensets, optional
        Advanced: provide a generator of candidate final sets C to restrict enumeration.
        Defaults to all subsets of V.

    Returns
    -------
    pmf : dict {tuple(counts_by_sorted_type_labels) -> probability}
        Joint PMF over counts by type (type order is meta['type_order']).
    meta : dict
        meta['type_order']: list of sorted unique type labels
        meta['P_exact']    : dict {C -> Pr(B^{(L)}_C)} if return_exact_sets=True
    """
    V = frozenset(V)
    if depth < 0:
        raise ValueError("depth must be >= 0")
    if not V:
        type_order = []
        meta = {"type_order": type_order}
        return {(): 1.0}, meta

    # resolve direct law
    if direct_law is None:
        if independent_p is None:
            raise ValueError("Provide either 'direct_law' or 'independent_p'.")
        direct_law = pi_from_independent_p(V, independent_p)

    # candidate final sets C (can be pruned by user-supplied generator)
    if candidate_sets is None:
        C_iter = _powerset(V)
    else:
        C_iter = candidate_sets(V, depth)

    be = BEKernel(Q)

    # exact-set probabilities
    P_exact = {}
    if depth == 0:
        # no propagation: exact final set equals direct-compromised set
        for C in C_iter:
            P_exact[C] = direct_law.get(C, 0.0)
    else:
        for C in C_iter:
            acc = 0.0
            for D0, pi in direct_law.items():
                if pi == 0.0:
                    continue
                g = be.gamma_multi(V, C, D0, depth)
                if g != 0.0:
                    acc += pi * g
            P_exact[C] = acc

    if normalize:
        tot = sum(P_exact.values())
        if tot > 0 and abs(tot - 1.0) > 1e-12:
            for C in P_exact:
                P_exact[C] /= tot

    # aggregate to type counts
    type_order = sorted({node_types[n] for n in V})
    idx_of = {lbl: i for i, lbl in enumerate(type_order)}
    pmf = {}
    for C, pr in P_exact.items():
        if pr == 0.0:
            continue
        counts = [0] * len(type_order)
        for n in C:
            counts[idx_of[node_types[n]]] += 1
        key = tuple(counts)
        pmf[key] = pmf.get(key, 0.0) + pr

    if normalize:
        s = sum(pmf.values())
        if s > 0 and abs(s - 1.0) > 1e-12:
            for k in list(pmf.keys()):
                pmf[k] = pmf[k] / s

    meta = {"type_order": type_order}
    if return_exact_sets:
        meta["P_exact"] = P_exact
    return pmf, meta
