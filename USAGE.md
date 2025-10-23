## Installation Options

### Option 1: Direct Installation from GitHub (Recommended for users)

```bash
pip install git+https://github.com/yourusername/BEALGO.git
```

### Option 2: Clone and Install Locally

```bash
git clone https://github.com/yourusername/BEALGO.git
cd BEALGO
pip install -e .
```

### Option 3: Download and Use Directly

If you don't want to install, you can simply download `be_joint_pmf.py` and import it in your project:

```bash
curl -O https://raw.githubusercontent.com/yourusername/BEALGO/main/be_joint_pmf.py
```

Then in your Python script:
```python
from be_joint_pmf import joint_pmf_by_types, pi_from_independent_p
```

## Quick Start

### Basic Example

```python
from be_joint_pmf import joint_pmf_by_types, pi_from_independent_p

# Define your network
V = [1, 2, 3, 4]  # 4 nodes
node_types = {1: 'A', 2: 'A', 3: 'B', 4: 'B'}  # 2 types

# Direct compromise probabilities
p = {1: 0.1, 2: 0.1, 3: 0.05, 4: 0.05}

# Edge probabilities (u->v with probability q_uv)
Q = {
    (1, 2): 0.2,
    (2, 1): 0.2,
    (3, 4): 0.15,
    (4, 3): 0.15,
}

# Compute joint PMF for 3-hop propagation
pmf, meta = joint_pmf_by_types(
    depth=3,
    V=V,
    node_types=node_types,
    Q=Q,
    independent_p=p
)

# Results
print("Type order:", meta["type_order"])
print("\nJoint PMF:")
for count_vector, probability in sorted(pmf.items()):
    if probability > 0.001:  # Only show non-negligible probabilities
        print(f"  P(X={count_vector}) = {probability:.6f}")
```

### Understanding the Output

The function returns:
- **pmf**: Dictionary mapping count tuples to probabilities
  - Keys are tuples `(count_type1, count_type2, ...)` 
  - Values are probabilities summing to 1.0
- **meta**: Dictionary with metadata
  - `type_order`: List showing the order of types in count tuples

For example, if `type_order = ['A', 'B']` and `pmf[(2, 1)] = 0.123`, this means:
- Probability is 0.123 that exactly 2 nodes of type 'A' and 1 node of type 'B' are compromised

### Advanced Example: Custom Direct-Entry Law

Instead of independent node probabilities, you can specify a custom distribution:

```python
# Explicit direct compromise scenarios
direct_law = {
    frozenset(): 0.7,           # 70% chance: no initial compromise
    frozenset([1]): 0.2,        # 20% chance: only node 1
    frozenset([1, 3]): 0.1,     # 10% chance: nodes 1 and 3
}

pmf, meta = joint_pmf_by_types(
    depth=2,
    V=V,
    node_types=node_types,
    Q=Q,
    direct_law=direct_law  # Use this instead of independent_p
)
```

### Getting Exact Set Probabilities

If you need probabilities for specific node combinations:

```python
pmf, meta = joint_pmf_by_types(
    depth=2,
    V=V,
    node_types=node_types,
    Q=Q,
    independent_p=p,
    return_exact_sets=True  # Enable exact set tracking
)

# Access exact probabilities
exact_probs = meta["P_exact"]
print(f"P(exactly nodes {{1,3}} compromised) = {exact_probs[frozenset([1, 3])]:.6f}")
```

## Real-World Applications

### 1. Cybersecurity: Attack Propagation

Model how malware spreads through a network:

```python
# Network: servers and workstations
servers = ['S1', 'S2', 'S3']
workstations = ['W1', 'W2', 'W3', 'W4']
V = servers + workstations

node_types = {node: 'Server' if node.startswith('S') else 'Workstation' 
              for node in V}

# Servers less likely to be directly compromised (better security)
p = {node: 0.01 if node.startswith('S') else 0.05 for node in V}

# Define network connections with propagation probabilities
Q = {
    ('S1', 'W1'): 0.3, ('S1', 'W2'): 0.3,
    ('S2', 'W2'): 0.3, ('S2', 'W3'): 0.3,
    # ... etc
}

pmf, meta = joint_pmf_by_types(depth=5, V=V, node_types=node_types, 
                                Q=Q, independent_p=p)
```

### 2. Epidemiology: Disease Spread

Track infection across population groups:

```python
# Population: adults and children
adults = list(range(1, 11))  # 10 adults
children = list(range(11, 21))  # 10 children
V = adults + children

node_types = {i: 'Adult' if i <= 10 else 'Child' for i in V}

# Initial infection probability
p = {i: 0.02 for i in V}  # 2% initially infected

# Contact probabilities (children interact more)
Q = {}
# ... define based on contact network
```

### 3. Infrastructure: Failure Propagation

Model cascading failures:

```python
# Power grid: generators and substations
generators = ['G1', 'G2']
substations = ['Sub1', 'Sub2', 'Sub3']
V = generators + substations

node_types = {node: 'Generator' if node.startswith('G') else 'Substation'
              for node in V}

# Failure probabilities
p = {node: 0.001 for node in V}  # Low base failure rate

# Cascading failure probabilities
Q = {('G1', 'Sub1'): 0.8, ('Sub1', 'Sub2'): 0.6, ...}
```

## Tips and Best Practices

1. **Start Small**: Test with 5-10 nodes first to understand behavior
2. **Sparse Q**: Only specify non-zero edge probabilities
3. **Type Aggregation**: Use meaningful types for easier interpretation
4. **Depth Selection**: Higher depth = more propagation rounds, but exponential complexity
5. **Normalization**: Keep `normalize=True` (default) to handle floating-point precision

## Troubleshooting

### Performance Issues
- **Problem**: Computation is slow
- **Solution**: Reduce network size or depth; algorithm complexity is exponential in |V|

### Memory Errors
- **Problem**: Out of memory
- **Solution**: The algorithm caches intermediate results. For large networks (>15 nodes), consider using type aggregation more aggressively

### Unexpected Probabilities
- **Problem**: Results don't sum to 1.0
- **Solution**: Ensure `normalize=True` (default). Check that Q values are valid probabilities [0,1]

### Import Errors
- **Problem**: `ModuleNotFoundError: No module named 'be_joint_pmf'`
- **Solution**: Ensure package is installed (`pip install -e .`) or `be_joint_pmf.py` is in your Python path

## API Reference

See the main README.md for detailed API documentation of:
- `joint_pmf_by_types()`
- `pi_from_independent_p()`

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For questions or issues:
- Open an issue on GitHub
- Contact the authors (see README.md)
