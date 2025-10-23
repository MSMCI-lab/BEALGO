# Examples

This directory contains example scripts demonstrating the BEALGO algorithm.

## example3_4.py

An 8-node network example with two node types (V1 and V2):
- Nodes 1-4: Type V1
- Nodes 5-8: Type V2
- 5-hop propagation model
- Shows basic usage of the algorithm

To run:
```bash
python example3_4.py
```

## Creating Your Own Example

Here's a template for creating your own example:

```python
from be_joint_pmf import joint_pmf_by_types

# 1. Define network structure
V = [1, 2, 3, 4]  # Your nodes
node_types = {1: 'TypeA', 2: 'TypeA', 3: 'TypeB', 4: 'TypeB'}

# 2. Define direct compromise probabilities
p = {1: 0.1, 2: 0.1, 3: 0.05, 4: 0.05}

# 3. Define edge probabilities (propagation)
Q = {
    (1, 2): 0.2,  # Node 1 can compromise node 2 with prob 0.2
    (2, 1): 0.15,
    # Add more edges as needed
}

# 4. Run the algorithm
pmf, meta = joint_pmf_by_types(
    depth=3,  # Number of propagation rounds
    V=V,
    node_types=node_types,
    Q=Q,
    independent_p=p
)

# 5. Analyze results
print("Type order:", meta["type_order"])
for counts, prob in sorted(pmf.items(), key=lambda x: -x[1]):
    if prob > 0.001:
        print(f"P(X={counts}) = {prob:.6f}")
```

## Additional Examples

### Example: Varying Depth

See how propagation affects the distribution:

```python
for depth in [0, 1, 2, 3, 5]:
    pmf, meta = joint_pmf_by_types(
        depth=depth,
        V=V,
        node_types=node_types,
        Q=Q,
        independent_p=p
    )
    print(f"\nDepth {depth}:")
    # Analyze results...
```

### Example: Comparing Scenarios

Compare different initial compromise scenarios:

```python
scenarios = {
    "Low Risk": {i: 0.01 for i in V},
    "Medium Risk": {i: 0.05 for i in V},
    "High Risk": {i: 0.10 for i in V},
}

for name, p_scenario in scenarios.items():
    pmf, meta = joint_pmf_by_types(
        depth=3, V=V, node_types=node_types,
        Q=Q, independent_p=p_scenario
    )
    print(f"\n{name} Scenario:")
    # Analyze results...
```
