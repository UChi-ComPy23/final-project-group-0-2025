import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import numpy as np
import matplotlib.pyplot as plt
from src.script_for_Langevin import UnadjustedLangevin

def gradV_tri(x, beta=1.0):
    # d/dx of ((x^2-1)(x^2-4)(x^2-9))/40
    a = x**2 - 1
    b = x**2 - 4
    c = x**2 - 9
    da = 2 * x
    db = 2 * x
    dc = 2 * x
    return beta * (da * b * c + a * db * c + a * b * dc) / 40.0


def pi0_sampler():
    return np.random.randn()



def test_unadjusted_langevin():
    """Test ULA with a tri-well target distribution using moment estimators."""
    beta = 1.0
    step_size = 0.01

    burn_in = 20_000
    N = 50_000

    # Moment estimators
    ULA_mean = UnadjustedLangevin(
        gradV=lambda x: gradV_tri(x, beta=beta),
        step_size=step_size,
        pi0_sampler=pi0_sampler,
        h=lambda x: x,
    )

    ULA_x2 = UnadjustedLangevin(
        gradV=lambda x: gradV_tri(x, beta=beta),
        step_size=step_size,
        pi0_sampler=pi0_sampler,
        h=lambda x: x**2,
    )

    ULA_x3 = UnadjustedLangevin(
        gradV=lambda x: gradV_tri(x, beta=beta),
        step_size=step_size,
        pi0_sampler=pi0_sampler,
        h=lambda x: x**3,
    )

    mean_est = ULA_mean.estimate(N, burn_in=burn_in)

    # Variance estimator
    h_var = lambda x: (x - mean_est)**2
    ULA_var = UnadjustedLangevin(
        gradV=lambda x: gradV_tri(x, beta=beta),
        step_size=step_size,
        pi0_sampler=pi0_sampler,
        h=h_var,
    )

    var_est = ULA_var.estimate(N, burn_in=burn_in)

    print("Unadjusted Langevin Test")
    print("E[X]      =", mean_est)
    print("E[X^2]    =", ULA_x2.estimate(N, burn_in=burn_in))
    print("Variance  =", var_est)
    print("E[X^3]    =", ULA_x3.estimate(N, burn_in=burn_in))


if __name__ == "__main__":
    test_unadjusted_langevin()