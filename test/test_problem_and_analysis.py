import numpy as np
import pytest
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal, assert_allclose

from src import problem
from src import analysis

# Setup for Problem Functions
BETA_TEST = 5.0

# Test 1: Complex Potential Functions (problem.py)

def test_complex_potential_symmetry():
    """
    Verifies that the complex potential V(x) is symmetric (V(x) = V(-x)).
    """
    x_val = 2.0
    v_positive = problem.complex_potential(x_val)
    v_negative = problem.complex_potential(-x_val)
    
    # Check V(x) = V(-x)
    assert_almost_equal(v_positive, v_negative)

def test_complex_potential_at_wells_and_barrier():
    """
    Verifies V(x) values at known locations (minima at +/- 1, 2, 3 and barrier at 0).
    The complex potential V(x) = (x^2 - 1)(x^2 - 4)(x^2 - 9) / 40 should be 0 at x=1, 2, 3.
    """
    # Minima are expected at x^2=1, 4, 9, so V(x) should be 0.
    assert_almost_equal(problem.complex_potential(1.0), 0.0)
    assert_almost_equal(problem.complex_potential(2.0), 0.0)
    assert_almost_equal(problem.complex_potential(3.0), 0.0)
    
    # Barrier at x=0: V(0) = (-1)(-4)(-9) / 40 = -36 / 40 = -0.9
    assert_almost_equal(problem.complex_potential(0.0), -0.9)


# Test 2: Generalized True Moment Calculation (problem.py)

def test_get_true_even_moment_with_simple_density():
    """
    Tests the generalized moment function using the simple double-well density.
    This ensures backward compatibility and correctness against known values (E[x^2] > 0.8 at beta=1).
    """
    beta_1 = 1.0
    Ex2_simple = problem.get_true_even_moment(problem.target_density, beta_1, 2)
    
    # Based on the experimental results (Table 5 in report), the true value is 0.8327 at beta=1.0, D=10
    assert_allclose(Ex2_simple, 0.8327, atol=0.001)

def test_get_true_odd_moment_returns_zero():
    """
    Verifies that the function correctly returns 0.0 for all odd moments (due to symmetry).
    """
    # Test with both densities
    assert_almost_equal(problem.get_true_even_moment(problem.target_density, BETA_TEST, 3), 0.0)
    assert_almost_equal(problem.get_true_even_moment(problem.complex_target_density, BETA_TEST, 1), 0.0)


# Test 3: Plotting Functions (analysis.py)

def test_plot_histogram_smoke_test():
    """
    Smoke test to ensure the generalized plot_histogram runs without crashing
    for both the simple and complex potentials.
    """
    # Generate mock data that spans a range (like an MCMC chain)
    mock_chain = np.random.uniform(-3.0, 3.0, 1000)
    
    fig, ax = plt.subplots(1, 1)

    # 1. Test with the Simple Potential (Default)
    analysis.plot_histogram(ax, mock_chain, title="Simple Test", beta=BETA_TEST)
    assert len(ax.lines) > 0 # Check that the density line was plotted

    # 2. Test with the Complex Potential (Explicitly Passed)
    ax.clear()
    analysis.plot_histogram(
        ax, 
        mock_chain, 
        title="Complex Test", 
        beta=BETA_TEST, 
        unnormalized_density_func=problem.complex_target_density
    )
    assert len(ax.lines) > 0 # Check that the density line was plotted
    
    plt.close(fig)