import pytest
import numpy as np
from src.parallel_tempering import *

def test_V():
    ''' V(x) = (x^2-1)^2'''
    assert V(1) == 0
    assert V(-1) == 0
    assert V(2) == 9
    assert V(3) == 64

def test_log_p():
    '''log(p(x,bata)) = -beta * V(x)'''
    assert log_p(1, 0.5) == 0
    assert log_p(-1, 0.5) == 0
    assert log_p(2, 0.5) == -4.5
    assert log_p(3, 0.5) == -32

Beta_max = 1.0
Beta_min = 0.01

def test_generate_temperature_ladder():
    '''
    Check the length of generated array
    Check the increasing order of generated array
    Check the equi-logspace
    '''
    assert len(generate_temp_ladder(10, Beta_min, Beta_max)) == 10
    assert len(generate_temp_ladder(20, Beta_max, Beta_min)) == 20

    Temps = generate_temp_ladder(10,Beta_min, Beta_max)
    for i in range(9):
        assert Temps[i] <= Temps[i+1]
    
    for i in range(8):
        assert np.allclose(Temps[i+1]/Temps[i], Temps[i+2]/Temps[i+1])

def test_parallel_tempering():
    '''
    test the length of generated array
    '''
    Betas = generate_temp_ladder(10,Beta_min, Beta_max)
    assert len(parallel_tempering(10, Betas)) == 10
    assert len(parallel_tempering(20, Betas)) == 20
        
    




