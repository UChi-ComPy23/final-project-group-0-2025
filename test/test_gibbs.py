import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_allclose

from src.metropolis import metropolis_hastings, tune_step_size, metropolis_step_1d
from src.gibbs import gibbs_sampler
from src import problem
from src.analysis import compute_expectation

# Setup for Universal M-H and Tuning Tests

# Use beta=5.0 and the original density for setup, testing universality
BETA_TEST = 5.0
DENSITY_FUNC_SIMPLE = lambda x, b=BETA_TEST: problem.target_density(x, b)
DENSITY_FUNC_COMPLEX = lambda x, b=BETA_TEST: problem.complex_target_density(x, b)

# Test 1: Universal M-H

def test_mh_universal_smoke():
    """
    Tests the refactored M-H function on the complex potential density.
    Verifies output length and runs without crashing.
    """
    n_steps = 1000
    chain = metropolis_hastings(
        n_steps, 0.0, 1.0, DENSITY_FUNC_COMPLEX, verbose=False
    )
    assert len(chain) == n_steps
    assert np.all(np.isfinite(chain))

def test_tuner_universal_runs():
    """
    Tests the refactored tuner works with the complex density function.
    """
    tuned_step = tune_step_size(
        initial_x=0.0,
        unnormalized_density_func=DENSITY_FUNC_COMPLEX,
        target_rate=0.4
    )
    # Check that a reasonable, positive step size was found
    assert tuned_step > 0.01

# Test 2: M-H Step (for Gibbs)

def test_mh_step_1d_acceptance():
    """
    Tests the single step helper function for basic functionality (acceptance).
    """
    # Use a small step size (0.1) and beta=1.0 to ensure reasonable acceptance
    new_x, accepted = metropolis_step_1d(
        current_x=0.0,
        step_size=0.1,
        unnormalized_density_func=problem.target_density,
        beta=1.0
    )
    # New x should be close to 0.0, and acceptance should be possible
    assert np.isfinite(new_x)
    
def test_mh_step_1d_zero_prob_start():
    """
    Tests safety against starting at a point where density is essentially zero.
    """
    # Start at a very high x where density is near zero
    new_x, accepted = metropolis_step_1d(
        current_x=10.0,
        step_size=1.0,
        unnormalized_density_func=problem.target_density,
        beta=1.0
    )
    # The M-H step should prevent crash and allow a move
    assert np.isfinite(new_x)

# Test 3: Gibbs Sampler (Core Logic)

def test_gibbs_smoke_shape():
    """
    Smoke test for the Gibbs sampler. Checks output shape and finiteness.
    """
    D = 3
    n_steps = 1000
    initial_x = np.array([1.0, 1.0, 1.0])
    
    chain_D = gibbs_sampler(
        D=D,
        n_steps=n_steps,
        beta=1.0,
        initial_x_vector=initial_x,
        step_size_1d=1.0,
        verbose=False
    )
    # Expected shape: (n_steps, D)
    assert chain_D.shape == (n_steps, D)
    assert np.all(np.isfinite(chain_D))

def test_gibbs_stuck_mode():
    """
    Verifies that the D-dimensional Gibbs sampler still exhibits the stuck mode 
    failure on the first component (x1) at high beta.
    """
    D = 10
    n_steps = 5000
    beta = 20.0       
    initial_x_value = 1.0 # Start near the positive mode
    step_size_1d = 0.1
    
    chain_D = gibbs_sampler(
        D=D,
        n_steps=n_steps,
        beta=beta,
        initial_x_vector=np.full(D, initial_x_value),
        step_size_1d=step_size_1d,
        verbose=False
    )
    
    # Analyze the first component (x1)
    chain_x1 = chain_D[:, 0]
    Ex1_estimate = compute_expectation(chain_x1, lambda x: x)
    
    # True mean is 0.0. If stuck near 1.0, the estimate should be close to 1.0.
    is_stuck = np.allclose(Ex1_estimate, 1.0, atol=0.2)     
    assert is_stuck, f"Gibbs (D={D}) should be stuck near 1.0. Got E[x1]={Ex1_estimate}"
    
    # Cannot return rate easily from gibbs_sampler, but can check the E[x1^2] is correct
    true_Ex2 = problem.get_true_even_moment(problem.target_density, beta, 2)
    Ex1_sq_estimate = compute_expectation(chain_x1, lambda x: x**2)
    assert_allclose(Ex1_sq_estimate, true_Ex2, atol=0.01) # Even moments are correct for the local mode
