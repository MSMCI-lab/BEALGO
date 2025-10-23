# BEALGO Quick Reference

## Installation

```bash
# From GitHub (once published)
pip install git+https://github.com/yourusername/BEALGO.git

# Or download single file
curl -O https://raw.githubusercontent.com/yourusername/BEALGO/main/be_joint_pmf.py
```

## Basic Usage

```python
from be_joint_pmf import joint_pmf_by_types

# Setup
V = [1, 2, 3, 4]  # Nodes
node_types = {1: 'A', 2: 'A', 3: 'B', 4: 'B'}  # Types
p = {1: 0.1, 2: 0.1, 3: 0.05, 4: 0.05}  # Direct compromise probs
Q = {(1, 2): 0.2, (2, 3): 0.15, ...}  # Edge propagation probs

# Compute
pmf, meta = joint_pmf_by_types(depth=3, V=V, node_types=node_types, 
                                Q=Q, independent_p=p)

# Results
print(meta["type_order"])  # ['A', 'B']
print(pmf[(2, 1)])  # P(2 type-A and 1 type-B compromised)
```

## Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `depth` | Propagation rounds (L ≥ 0) | `depth=3` |
| `V` | List of nodes | `[1,2,3,4]` or `['n1','n2']` |
| `node_types` | Node → type mapping | `{1:'A', 2:'B'}` |
| `Q` | Edge probabilities | `{(1,2): 0.3}` means node 1→2 with prob 0.3 |
| `independent_p` | Per-node direct probs | `{1: 0.1, 2: 0.2}` |
| `direct_law` | Custom direct-entry dist | `{frozenset([1]): 0.5}` |

## Return Values

```python
pmf, meta = joint_pmf_by_types(...)
```

- **pmf**: `dict` mapping `(count_type1, count_type2, ...)` → probability
- **meta**: `dict` with:
  - `"type_order"`: list of type labels in order
  - `"P_exact"`: (if `return_exact_sets=True`) exact set probabilities

## Common Patterns

### Multiple Depths
```python
for L in [0, 1, 2, 3, 5]:
    pmf, meta = joint_pmf_by_types(depth=L, V=V, node_types=node_types,
                                    Q=Q, independent_p=p)
    # Analyze how distribution changes with depth
```

### Expected Values
```python
pmf, meta = joint_pmf_by_types(...)
types = meta["type_order"]

# Expected number of type 0 compromised
E_X0 = sum(counts[0] * prob for counts, prob in pmf.items())
print(f"E[{types[0]}] = {E_X0:.4f}")
```

### Probability of at least k compromised
```python
# P(X_A >= 2) - at least 2 of type A
type_idx = meta["type_order"].index('A')
prob = sum(p for counts, p in pmf.items() if counts[type_idx] >= 2)
```

### Custom Direct-Entry Scenarios
```python
# Scenario 1: Targeted attack (node 1 always compromised)
direct_law = {frozenset([1]): 1.0}

# Scenario 2: Random single node
direct_law = {frozenset([i]): 1/len(V) for i in V}

pmf, meta = joint_pmf_by_types(depth=2, V=V, node_types=node_types,
                                Q=Q, direct_law=direct_law)
```

## Tips

1. **Sparse Q**: Only specify non-zero edges: `Q = {(u,v): q_uv for (u,v) in edges}`
2. **Node Types**: Can be any hashable: strings, ints, tuples
3. **Scalability**: Practical for ~10-15 nodes; exponential complexity
4. **Normalization**: Keep default `normalize=True` to handle floating-point precision
5. **Depth=0**: Returns direct-entry distribution only (no propagation)

## API Summary

### Main Functions

```python
joint_pmf_by_types(depth, V, node_types, Q, 
                   independent_p=None, direct_law=None,
                   return_exact_sets=False, normalize=True,
                   candidate_sets=None)
```
Compute exact joint PMF over type counts.

```python
pi_from_independent_p(V, p)
```
Convert independent node probabilities to direct-entry law.

## Examples

- **example3_4.py**: 8-node network with 2 types
- **USAGE.md**: Detailed usage guide with applications
- **EXAMPLES.md**: More example patterns

## Testing

```bash
python3 test_bealgo.py
```

## Documentation

- **README.md**: Overview and features
- **USAGE.md**: Detailed usage and applications
- **EXAMPLES.md**: Example patterns
- **PUBLISHING_GUIDE.md**: How to publish to GitHub

## Support

- GitHub Issues: Report bugs or request features
- Examples: Check example scripts for common use cases
- Tests: Run tests to verify installation

---
**Version**: 0.1.0 | **License**: MIT | **Python**: ≥3.7
