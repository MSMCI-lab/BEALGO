"""
Simple tests for BEALGO to verify functionality.
Run with: python test_bealgo.py
"""

from be_joint_pmf import joint_pmf_by_types, pi_from_independent_p


def test_basic_functionality():
    """Test basic functionality with a simple 2-node network."""
    print("Test 1: Basic 2-node network...")
    
    V = [1, 2]
    node_types = {1: 'A', 2: 'B'}
    p = {1: 0.1, 2: 0.2}
    Q = {(1, 2): 0.3}
    
    pmf, meta = joint_pmf_by_types(
        depth=1,
        V=V,
        node_types=node_types,
        Q=Q,
        independent_p=p
    )
    
    # Check that probabilities sum to 1
    total = sum(pmf.values())
    assert abs(total - 1.0) < 1e-10, f"Probabilities don't sum to 1: {total}"
    
    # Check that type order is correct
    assert meta["type_order"] == ['A', 'B'], f"Type order incorrect: {meta['type_order']}"
    
    print("  ✓ Probabilities sum to 1.0")
    print("  ✓ Type order correct")
    print("  ✓ Test passed!\n")


def test_zero_depth():
    """Test depth=0 (no propagation)."""
    print("Test 2: Zero depth (no propagation)...")
    
    V = [1, 2, 3]
    node_types = {1: 'A', 2: 'A', 3: 'B'}
    p = {1: 0.1, 2: 0.1, 3: 0.1}
    Q = {(1, 2): 0.5, (2, 3): 0.5}
    
    pmf, meta = joint_pmf_by_types(
        depth=0,
        V=V,
        node_types=node_types,
        Q=Q,
        independent_p=p
    )
    
    total = sum(pmf.values())
    assert abs(total - 1.0) < 1e-10, f"Probabilities don't sum to 1: {total}"
    
    print("  ✓ Zero depth works correctly")
    print("  ✓ Test passed!\n")


def test_custom_direct_law():
    """Test with custom direct-entry law."""
    print("Test 3: Custom direct-entry law...")
    
    V = [1, 2]
    node_types = {1: 'A', 2: 'B'}
    Q = {(1, 2): 0.5}
    
    # Custom direct law: 50% chance of node 1, 50% chance of nothing
    direct_law = {
        frozenset(): 0.5,
        frozenset([1]): 0.5,
    }
    
    pmf, meta = joint_pmf_by_types(
        depth=1,
        V=V,
        node_types=node_types,
        Q=Q,
        direct_law=direct_law
    )
    
    total = sum(pmf.values())
    assert abs(total - 1.0) < 1e-10, f"Probabilities don't sum to 1: {total}"
    
    print("  ✓ Custom direct law works")
    print("  ✓ Test passed!\n")


def test_exact_sets():
    """Test return_exact_sets option."""
    print("Test 4: Exact set probabilities...")
    
    V = [1, 2]
    node_types = {1: 'A', 2: 'A'}
    p = {1: 0.5, 2: 0.5}
    Q = {}
    
    pmf, meta = joint_pmf_by_types(
        depth=0,
        V=V,
        node_types=node_types,
        Q=Q,
        independent_p=p,
        return_exact_sets=True
    )
    
    assert "P_exact" in meta, "P_exact not in metadata"
    
    # For independent nodes with depth=0, we should have 4 exact sets
    exact_probs = meta["P_exact"]
    assert len(exact_probs) == 4, f"Should have 4 exact sets, got {len(exact_probs)}"
    
    total = sum(exact_probs.values())
    assert abs(total - 1.0) < 1e-10, f"Exact probabilities don't sum to 1: {total}"
    
    print("  ✓ Exact sets returned correctly")
    print("  ✓ Test passed!\n")


def test_example_network():
    """Test with the example from example3_4.py."""
    print("Test 5: Example 8-node network...")
    
    V = list(range(1, 9))
    node_types = {i: ('V1' if i <= 4 else 'V2') for i in V}
    p = {i: (0.10 if i <= 4 else 0.05) for i in V}
    
    # Simplified Q for testing
    Q = {
        (1, 2): 0.15, (1, 5): 0.2,
        (2, 1): 0.15, (2, 6): 0.2,
        (3, 4): 0.15, (3, 7): 0.2,
        (4, 3): 0.15, (4, 8): 0.2,
        (5, 1): 0.2, (5, 6): 0.1,
        (6, 2): 0.2, (6, 5): 0.1,
        (7, 3): 0.2, (7, 8): 0.1,
        (8, 4): 0.2, (8, 7): 0.1,
    }
    
    pmf, meta = joint_pmf_by_types(
        depth=2,
        V=V,
        node_types=node_types,
        Q=Q,
        independent_p=p
    )
    
    total = sum(pmf.values())
    assert abs(total - 1.0) < 1e-10, f"Probabilities don't sum to 1: {total}"
    
    # Check that we have reasonable distribution
    assert len(pmf) > 0, "PMF is empty"
    assert meta["type_order"] == ['V1', 'V2'], f"Type order incorrect: {meta['type_order']}"
    
    print("  ✓ 8-node network computes correctly")
    print("  ✓ Test passed!\n")


def test_pi_from_independent_p():
    """Test pi_from_independent_p utility function."""
    print("Test 6: pi_from_independent_p function...")
    
    V = [1, 2]
    p = {1: 0.3, 2: 0.4}
    
    pi = pi_from_independent_p(V, p)
    
    # Should have 2^2 = 4 entries
    assert len(pi) == 4, f"Should have 4 entries, got {len(pi)}"
    
    # Check specific probabilities
    prob_none = pi[frozenset()]
    expected_none = (1 - 0.3) * (1 - 0.4)
    assert abs(prob_none - expected_none) < 1e-10, "Probability calculation incorrect"
    
    total = sum(pi.values())
    assert abs(total - 1.0) < 1e-10, f"Probabilities don't sum to 1: {total}"
    
    print("  ✓ pi_from_independent_p works correctly")
    print("  ✓ Test passed!\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running BEALGO Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_basic_functionality,
        test_zero_depth,
        test_custom_direct_law,
        test_exact_sets,
        test_example_network,
        test_pi_from_independent_p,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ Test failed with error: {e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✅ All tests passed! BEALGO is working correctly.\n")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed. Please check the errors above.\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
