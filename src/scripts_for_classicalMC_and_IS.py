import numpy as np
from scipy.stats import norm

class ClassicalMonteCarlo:
    """
    Classical Monte Carlo integration: I_f[h] = (1/N) * sum(h(X^(n))), where X^(n) ~ f  i.i.d.

    Inputs
    sampler: Returns N i.i.d. samples from the target distribution f.
    h: Test function to apply to samples.
    """

    def __init__(self, sampler, h):
        self.sampler = sampler
        self.h = h

    def estimate(self, N):
        X = self.sampler(N)
        values = self.h(X)
        return np.mean(values)

class ImportanceSampling:
    """
    Self-normalized Importance Sampling estimator for expectations under an unnormalized target density p(x).

    Inputs
    proposal_sampler : Function that takes N and returns N samples from proposal q(x).
    proposal_pdf: Probability density function q(x) evaluated pointwise.
    target_unnorm: Unnormalized density p(x).
    h: Test function whose expectation E_p[h(X)] is desired.
    """

    def __init__(self, proposal_sampler, proposal_pdf, target_unnorm, h):
        self.q_sampler = proposal_sampler
        self.q_pdf = proposal_pdf
        self.p_unnorm = target_unnorm
        self.h = h

    def estimate(self, N):
        y = self.q_sampler(N)
        w = self.p_unnorm(y) / self.q_pdf(y)

        h_vals = self.h(y)
        return np.sum(h_vals * w) / np.sum(w)


# testing
if __name__ == "__main__":

    # test for classical mote carlo
    # sample is normal distribution
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
    # varince
    h_var = lambda x: (x - mean_est)**2
    mc_var = ClassicalMonteCarlo(sampler, h_var)
    var_est = mc_var.estimate(N)

    print("Classical Monte Carlo Results")
    print("E[X] = ", mean_est)
    print("E[X^2] = ", m2_est)
    print("variance: ", var_est)
    

    # test for importance sampling
    # p(x) = exp(-beta*(x^2 - 1)^2)
    beta = 1.0
    target_unnorm = lambda x: np.exp(-beta * (x**2 - 1)**2)

    # proposed normal distribution
    proposal_mu = 0.0
    proposal_sigma = 2.0

    proposal_sampler = lambda N: np.random.normal(proposal_mu, proposal_sigma, size=N)
    proposal_pdf = lambda x: norm.pdf(x, proposal_mu, proposal_sigma)

    N = 50000

    # Estimate E[X], E[X^2], E[X^3]
    IS_mean = ImportanceSampling(proposal_sampler, proposal_pdf, target_unnorm, h=lambda x: x)
    IS_x2   = ImportanceSampling(proposal_sampler, proposal_pdf, target_unnorm, h=lambda x: x**2)
    IS_x3   = ImportanceSampling(proposal_sampler, proposal_pdf, target_unnorm, h=lambda x: x**3)

    IS_mean_est = IS_mean.estimate(N)
    # varince
    h_var = lambda x: (x - IS_mean_est)**2
    mc_var = ImportanceSampling(proposal_sampler, proposal_pdf, target_unnorm, h_var)
    IS_var_est = mc_var.estimate(N)
    print("Importance Sampling Results")
    print("E[X] = ", IS_mean_est)
    print("E[X^2] = ", IS_x2.estimate(N))
    print("variance: ", IS_var_est)
    print("E[X^3] = ", IS_x3.estimate(N))
