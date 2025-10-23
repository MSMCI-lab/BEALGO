# main_example.py
from be_joint_pmf import joint_pmf_by_types, pi_from_independent_p

# Nodes & types
V = list(range(1, 9))                           # 1..8
node_types = {i: ('V1' if i <= 4 else 'V2') for i in V}

# Direct-entry: independent per node
p = {i: (0.10 if i <= 4 else 0.05) for i in V}  # Example 3.4
# Alternatively: direct_law = {...}  # explicit π(D0)

# Edge probabilities Q (only nonzero edges needed)
Q = {}
mat = [
    [0,0.15,0,0.15,0.2,0,0.2,0],
    [0.15,0,0.15,0,0.2,0.2,0,0],
    [0,0.15,0,0.15,0,0.2,0,0.2],
    [0.15,0,0.15,0,0,0,0.2,0.2],
    [0.2,0.2,0,0,0,0.1,0.1,0.1],
    [0,0.2,0.2,0,0.1,0,0.1,0.1],
    [0.2,0,0,0.2,0.1,0.1,0,0.1],
    [0,0,0.2,0.2,0.1,0.1,0.1,0],
]
for i in range(8):
    for j in range(8):
        if i != j and mat[i][j] != 0:
            Q[(i+1, j+1)] = mat[i][j]

# Compute joint PMF for depth L=5
pmf, meta = joint_pmf_by_types(
    depth=5,
    V=V,
    node_types=node_types,
    Q=Q,
    independent_p=p,          # or direct_law=...
    return_exact_sets=False,  # True if you want Pr(B_C) for each exact set C
)

print("Type order:", meta["type_order"])  # ['V1','V2']
# Example: probability that x1=2, x2=3
print("Pr[X1=2, X2=3] =", pmf.get((1,3), 0.0))
