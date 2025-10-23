# BEALGO: Backward Elimination Algorithm for Joint PMF Computation

This repository contains a Python implementation of the Backward Elimination (BE) algorithm for computing exact joint probability mass functions (PMF) in multi-hop network propagation models.

## Overview

The BEALGO algorithm computes the exact joint distribution of infection/compromise counts by node types in networks where:
- Nodes can be directly compromised with given probabilities
- Compromised nodes can spread to neighbors with edge-specific probabilities
- Propagation occurs over multiple rounds (L-hop model)
- Nodes are categorized into different types for analysis

This is particularly useful in:
- **Network Security**: Modeling malware/attack propagation
- **Epidemiology**: Disease spread modeling
- **Social Networks**: Information diffusion analysis
- **Infrastructure**: Failure propagation in systems

## Key Features

- **Exact Computation**: Provides mathematically precise joint PMF (no approximation)
- **Type-based Aggregation**: Groups nodes by types for scalable analysis
- **Flexible Input**: Supports both independent node probabilities and custom direct-entry laws
- **Memory Efficient**: Uses memoization to optimize recursive computations
- **L-hop Propagation**: Models multi-round spreading dynamics

## Installation

No external dependencies required - uses only Python standard library.

```bash
git clone <repository-url>
cd BEALGO
```

## Quick Start

```python
from be_joint_pmf import joint_pmf_by_types, pi_from_independent_p

# Define network
V = list(range(1, 9))  # Nodes 1-8
node_types = {i: ('V1' if i <= 4 else 'V2') for i in V}

# Direct compromise probabilities
p = {i: (0.10 if i <= 4 else 0.05) for i in V}

# Edge probabilities (only specify non-zero edges)
Q = {
    (1, 2): 0.15, (1, 4): 0.15, (1, 5): 0.2, (1, 7): 0.2,
    (2, 1): 0.15, (2, 3): 0.15, (2, 5): 0.2, (2, 6): 0.2,
    # ... (see example3_4.py for complete specification)
}

# Compute joint PMF for 5-hop propagation
pmf, meta = joint_pmf_by_types(
    depth=5,
    V=V,
    node_types=node_types,
    Q=Q,
    independent_p=p
)

print("Type order:", meta["type_order"])  # ['V1', 'V2']
print("P(X1=2, X2=3):", pmf.get((2, 3), 0.0))
```

## Core Functions

### `joint_pmf_by_types()`
Main function that computes the joint PMF over type-count vectors.

**Parameters:**
- `depth`: Number of propagation rounds (L ≥ 0)
- `V`: Iterable of nodes (any hashable type)
- `node_types`: Dict mapping nodes to type labels
- `Q`: Dict of edge probabilities `{(u,v): q_uv}`
- `independent_p`: Dict of per-node direct probabilities (alternative to `direct_law`)
- `direct_law`: Explicit direct-entry distribution (alternative to `independent_p`)

**Returns:**
- `pmf`: Dict mapping count tuples to probabilities
- `meta`: Metadata including type ordering

### `pi_from_independent_p()`
Utility function to convert independent node probabilities to direct-entry law.

## Algorithm Details

The implementation follows a two-stage approach:

1. **BE Kernels (Algorithm 1)**: Compute transition probabilities γ^(r)_U(C|D₀) using recursive backward elimination
2. **Joint PMF (Algorithm 2)**: Aggregate exact set probabilities into type-count distributions

### Mathematical Foundation

For depth L propagation:
- **Direct Entry**: Nodes compromised initially with probability π(D₀)
- **Propagation**: Multi-round spreading with edge probabilities q_uv
- **Final State**: Joint distribution over compromise counts by node type

The exact PMF is computed as:
```
P(X = x) = Σ_{C: count(C)=x} P(B^(L)_C)
```
where `B^(L)_C` is the event that exactly set C is compromised after L rounds.

## File Structure

```
BEALGO/
├── be_joint_pmf.py    # Core algorithm implementation
├── example3_4.py      # Example usage (8-node network)
├── README.md          # This file
└── __pycache__/       # Python cache directory
```

## Examples

See `example3_4.py` for a complete working example with an 8-node network divided into two types (V1: nodes 1-4, V2: nodes 5-8).

## Performance Notes

- **Complexity**: Exponential in |V| due to exact enumeration
- **Optimization**: Memoization reduces redundant computations
- **Scalability**: Practical for networks with ~10-15 nodes
- **Memory**: Caches intermediate results for efficiency

## Advanced Usage

### Custom Direct-Entry Laws
```python
# Instead of independent_p, provide explicit distribution
direct_law = {
    frozenset(): 0.8,        # No initial compromise
    frozenset([1]): 0.1,     # Only node 1
    frozenset([1,2]): 0.1,   # Nodes 1 and 2
}

pmf, meta = joint_pmf_by_types(
    depth=3, V=V, node_types=node_types, Q=Q,
    direct_law=direct_law
)
```

### Exact Set Probabilities
```python
pmf, meta = joint_pmf_by_types(
    depth=3, V=V, node_types=node_types, Q=Q,
    independent_p=p,
    return_exact_sets=True  # Include P(B_C) for each set C
)

exact_probs = meta["P_exact"]
print("P(exactly {1,3} compromised):", exact_probs[frozenset([1,3])])
```

## Contributing

This is a research collaboration between multiple institutions. For questions or contributions, please contact the authors.

## License

[Please specify license information]

## Citation

If you use this code in your research, please cite:

```bibtex
[Citation information to be added]
```

## Authors

- [Author names and affiliations to be added]

## References

[Relevant papers and theoretical background to be added]