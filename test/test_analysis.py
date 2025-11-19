import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from src.analysis import compute_expectation, calculate_metrics

def test_compute_expectation_unweighted():
    """
    Test standard mean calculation.
    """
    samples = np.array([1, 2, 3, 4, 5])
    f_x = lambda x: x
    
    # Test: First moment E[x]
    expected = 3.0
    result = compute_expectation(samples, f_x, weights=None)
    assert_almost_equal(result, expected)

    # Test 2: Second moment E[x^2]
    f_x2 = lambda x: x**2 # [1, 4, 9, 16, 25] -> mean is 11.0
    expected_x2 = 11.0
    result_x2 = compute_expectation(samples, f_x2, weights=None)
    assert_almost_equal(result_x2, expected_x2)

def test_compute_expectation_weighted():
    """
    Test weighted mean calculation.
    """
    f_x = lambda x: x
    
    # Test: Simple weights (should equal unweighted mean)
    samples_even = np.array([1, 2, 3])
    weights_even = np.array([1, 1, 1])
    expected_even = 2.0
    result_even = compute_expectation(samples_even, f_x, weights=weights_even)
    assert_almost_equal(result_even, expected_even)
    
    # Test: Complex weights (35/13)
    # Calculation: (1*1 + 2*2 + 3*10) / (1 + 2 + 10) = 35 / 13
    samples_uneven = np.array([1, 2, 3])
    weights_uneven = np.array([1, 2, 10])
    expected_uneven = 35.0 / 13.0
    result_uneven = compute_expectation(samples_uneven, f_x, weights=weights_uneven)
    assert_almost_equal(result_uneven, expected_uneven)

def test_calculate_metrics():
    """
    Test bias and variance calculation.
    """
    # Case 1: Zero bias
    estimates_unbiased = [1.1, 0.9, 1.0] # Mean is 1.0
    true_value = 1.0
    
    # Expected variance: np.var([1.1, 0.9, 1.0], ddof=1) = 0.01
    results = calculate_metrics(estimates_unbiased, true_value)
    assert_almost_equal(results['bias'], 0.0)
    assert_almost_equal(results['variance'], 0.01)

    # Case 2: Non-zero bias
    estimates_biased = [1.1, 1.2, 1.3] # Mean is 1.2
    true_value = 1.0
    
    # Expected bias: |1.2 - 1.0| = 0.2
    # Expected variance: np.var([1.1, 1.2, 1.3], ddof=1) = 0.01
    results_biased = calculate_metrics(estimates_biased, true_value)
    assert_almost_equal(results_biased['bias'], 0.2)
    assert_almost_equal(results_biased['variance'], 0.01)