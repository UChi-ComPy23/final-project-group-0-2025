import pytest
import numpy as np
from src.parallel_tempering import *
from scipy.integrate import quad
from scipy.stats import kstest

def test_V1():
    ''' V(x) = (x^2-1)^2'''
    assert V1(1) == 0
    assert V1(-1) == 0
    assert V1(2) == 9
    assert V1(3) == 64

def test_log_p_V1():
    '''log(p(x,bata)) = -beta * V(x)'''
    assert log_p(1, 0.5, V1) == 0
    assert log_p(-1, 0.5, V1) == 0
    assert log_p(2, 0.5, V1) == -4.5
    assert log_p(3, 0.5, V1) == -32

Beta_max = 1.0
Beta_min = 0.01

def test_generate_betas():
    '''
    Check the length of generated array
    Check the increasing order of generated array
    Check the equi-logspace
    '''
    assert len(generate_betas(10, Beta_min, Beta_max)) == 10
    assert len(generate_betas(20, Beta_max, Beta_min)) == 20

    Temps = generate_betas(10,Beta_min, Beta_max)
    for i in range(9):
        assert Temps[i] <= Temps[i+1]
    
    for i in range(8):
        assert np.allclose(Temps[i+1]/Temps[i], Temps[i+2]/Temps[i+1])

def test_parallel_tempering_V1():
    '''
    test the length of generated array
    test the distribution of generated X_i's
    '''
    Betas = generate_betas(n=40, beta_min=0.05, beta_max=20)

    # use Kolmogorov-Smornov test to check whether generate
    # X_i with correct distribution
    
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
    
    xs = parallel_tempering(10000,1000,Betas,3,V1)
    _, pvalue = kstest(xs, cdf_v, args=(20.0,))

    # p_value > 0.05 => we fail to reject H_0: samples are 
    # statistically good at fitting the target pdf.
    assert pvalue > 0.05

def test_parallel_tempering_V2():
    '''
    test the distribution of generated X_i's
    '''
    Betas = generate_betas(n=75, beta_min=0.001, beta_max=20)

    # use Kolmogorov-Smornov test to check whether generate
    # X_i with correct distribution
    
    # This is e^{-beta*V(x)}
    def pdf_raw(x):
        return np.exp(log_p(x, 20.0, V2))
    # get the denominator, which is a normalizer
    denominator, _ = quad(pdf_raw, -np.inf, np.inf)
    
    def pdf(x, beta=20.0): 
        return np.exp(log_p(x,beta,V2))/denominator
    
    # vectorize the pdf because quad cannot accept x as an array

    pdf_v = np.vectorize(pdf)
    
    def cdf(x, beta=20.0):
        outcome, _ = quad(pdf_v, -np.inf, x, args=(beta,))
        return outcome
    
    # vectorize the pdf because quad cannot accept x as an array
    cdf_v = np.vectorize(cdf)
    
    xs = parallel_tempering(10000,1000,Betas,5,V2)
    _, pvalue = kstest(xs, cdf_v, args=(20.0,))


    # p_value > 0.05 => we fail to reject H_0: samples are 
    # statistically good at fitting the target pdf.
    assert pvalue > 0.05
