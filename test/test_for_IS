import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.scripts_for_classicalMC_and_IS import ImportanceSampling

import numpy as np
from scipy.stats import norm

def test_importance_sampling():
    """Test Importance Sampling with a double-well target distribution."""
    beta = 1.0
    target_unnorm = lambda x: np.exp(-beta * (x**2 - 1)**2)

    # Proposal distribution q = N(0,2^2)
    proposal_mu = 0.0
    proposal_sigma = 2.0
    proposal_sampler = lambda N: np.random.normal(proposal_mu, proposal_sigma, size=N)
    proposal_pdf = lambda x: norm.pdf(x, proposal_mu, proposal_sigma)

    N = 50_000

    # Moment estimators
    IS_mean = ImportanceSampling(proposal_sampler, proposal_pdf, target_unnorm, h=lambda x: x)
    IS_x2   = ImportanceSampling(proposal_sampler, proposal_pdf, target_unnorm, h=lambda x: x**2)
    IS_x3   = ImportanceSampling(proposal_sampler, proposal_pdf, target_unnorm, h=lambda x: x**3)

    mean_est = IS_mean.estimate(N)

    # Variance estimator
    h_var = lambda x: (x - mean_est)**2
    IS_var = ImportanceSampling(proposal_sampler, proposal_pdf, target_unnorm, h_var)
    var_est = IS_var.estimate(N)

    print("=== Importance Sampling Test ===")
    print("E[X] =", mean_est)
    print("E[X^2] =", IS_x2.estimate(N))
    print("Variance =", var_est)
    print("E[X^3] =", IS_x3.estimate(N))

if __name__ == "__main__":
    test_importance_sampling()
