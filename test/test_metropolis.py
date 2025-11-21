import numpy as np
import pytest

from src.metropolis import metropolis_hastings, tune_step_size
from src.analysis import compute_expectation
from src import problem

def test_metropolis_hastings_smoke_test():
    """
    Smoke test to verify M-H runs without crashing and returns the correct length.
    Uses an easy beta (1.0) for fast execution.
    """
    n_steps = 1000
    beta = 1.0  # Use an easy beta
    initial_x = 0.0
    step_size = 1.0
    burn_in_frac = 0.1
    
    # Run without the acceptance rate
    chain = metropolis_hastings(n_steps, beta, initial_x, step_size, 
                                burn_in=burn_in_frac, verbose=False)
    
    # Check output length equals requested samples (n_steps)
    expected_len = n_steps
    assert len(chain) == expected_len
    
    # 2. Check all chain values are finite (no NaNs or inf)
    assert np.all(np.isfinite(chain))

def test_metropolis_hastings_high_beta():
    """
    Verifies the MCMC failure mode at high beta. 
    The chain should get stuck in one mode, resulting in high bias (E[x] != 0).
    """
    # Use parameters that guarantee failure (high beta = deep well, small step)
    n_steps = 5000
    beta = 20.0       
    initial_x = 1.0 # Start near a peak
    step_size = 0.1    
    burn_in_steps = 1000
    
    chain = metropolis_hastings(n_steps, beta, initial_x, step_size, 
                                burn_in=burn_in_steps, verbose=False)
    
    # Check length
    assert len(chain) == n_steps
    
    # Check E[x^2] (Even Moment) should be correct
    f_x2 = lambda x: x**2
    Ex2_estimate = compute_expectation(chain, f_x2)
    
    true_Ex2 = problem.get_true_even_moment(beta, 2)
    assert np.allclose(Ex2_estimate, true_Ex2, atol=0.2) 

    # Check: E[x] (Odd Moment) should be incorrect (biased)
    f_x = lambda x: x
    Ex_estimate = compute_expectation(chain, f_x)
    print(f"\nDEBUG (high_beta): E[x] estimate = {Ex_estimate}\n")
    
    # Check if it's stuck near 1 OR -1
    is_stuck = np.allclose(Ex_estimate, 1.0, atol=0.2) or \
               np.allclose(Ex_estimate, -1.0, atol=0.2)
               
    assert is_stuck, "At high beta, M-H should get stuck near 1 or -1"

def test_tune_step_size():
    """
    Test the tuner function to see if it runs and returns a plausible (positive) step size.
    """
    # Use a medium beta so the tuner actually has to do some work
    tuned_step = tune_step_size(beta=5.0, initial_x=0.0, target_rate=0.4,
                               n_tune_steps=1000)
    
    # The step size should be a reasonable positive number
    assert tuned_step > 0
    assert tuned_step < 5.0
