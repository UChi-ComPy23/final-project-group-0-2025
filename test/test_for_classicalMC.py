import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.scripts_for_classicalMC_and_IS import ClassicalMonteCarlo
import numpy as np

def test_classical_monte_carlo():
    """Test Classical Monte Carlo on N(0,1) with moments."""
    sampler = lambda N: np.random.normal(loc=0.0, scale=1.0, size=N)
    N = 100_000

    # E[X]
    h_mean = lambda x: x
    mc_mean = ClassicalMonteCarlo(sampler, h_mean)
    mean_est = mc_mean.estimate(N)

    # E[X^2]
    h_second = lambda x: x**2
    mc_second = ClassicalMonteCarlo(sampler, h_second)
    m2_est = mc_second.estimate(N)

    # Var(X) = E[(X - E[X])^2]
    h_var = lambda x: (x - mean_est)**2
    mc_var = ClassicalMonteCarlo(sampler, h_var)
    var_est = mc_var.estimate(N)

    print("=== Classical Monte Carlo Test ===")
    print("E[X] =", mean_est)
    print("E[X^2] =", m2_est)
    print("Variance =", var_est)

if __name__ == "__main__":
    test_classical_monte_carlo()
