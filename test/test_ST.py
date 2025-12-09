import pytest
import numpy as np
from src.simulated_tempering import *
from scipy.integrate import quad
from scipy.stats import kstest

def test_simulated_tempering_V1():
    betas = generate_betas(10, beta_min = 0.01, beta_max=20)
    # use KS test to check the distribution of generated samples
    # This is e^{-beta*V(x)}
    def pdf_raw(x):
        return np.exp(log_p(x, 20.0, V1))
    # get the denominator, which is a normalizer
    denominator, _ = quad(pdf_raw, -np.inf, np.inf)
    
    def pdf(x, beta=20.0): 
        return np.exp(log_p(x,beta,V1))/denominator
    
    # vectorize the pdf because quad cannot accept x as an array

    pdf_v = np.vectorize(pdf)
    
    def cdf(x, beta=20.0):
        outcome, _ = quad(pdf_v, -np.inf, x, args=(beta,))
        return outcome
    
    # vectorize the pdf because quad cannot accept x as an array
    cdf_v = np.vectorize(cdf)
    
    xs, beta_idx = simulated_tempering(10000,1000,betas,3,V1)
    target_xs = xs[beta_idx == 0]
    _, pvalue = kstest(target_xs, cdf_v, args=(20.0,))

    # p_value > 0.05 => we fail to reject H_0: samples are 
    # statistically good at fitting the target pdf.
    assert pvalue > 0.05

