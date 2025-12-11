import numpy as np
from scipy.integrate import quad

def potential(x):
    """
    Calculates the 1D double-well potential V(x) = (x^2 - 1)^2
    """
    return (x**2 - 1)**2

def target_density(x, beta):
    """
    Returns the unnormalized target density, p(x; beta) = exp(-beta * V(x))
    """
    return np.exp(-beta * potential(x))

# Ground Truth Values: by symmetry, all odd moments are 0
TRUE_Ex = 0.0
TRUE_Ex3 = 0.0
TRUE_Ex5 = 0.0 

# OLD def get_true_even_moment(beta, power) commented out due to adjustment in final project version
# def get_true_even_moment(beta, power):
#     """
#     Calculates the exact value for E[x^power] using numerical integration (quadrature).
#     This serves as the benchmark/ground truth for the sampling methods.
#     """
#     if power % 2 != 0:
#         return 0.0 # All odd moments are 0
        
#     # Integrand for the numerator: x^n * p(x)
#     numerator_func = lambda x: (x**power) * target_density(x, beta)
    
#     # Integrand for the denominator (Normalization constant Z): p(x)
#     denominator_func = lambda x: target_density(x, beta)
    
#     # Perform the integration
#     # quad(function, lower_limit, upper_limit)
#     # quad returns (result, error), we only need the result
#     numerator, _ = quad(numerator_func, -np.inf, np.inf)
#     denominator, _ = quad(denominator_func, -np.inf, np.inf)
    
#     if denominator == 0:
#         return np.nan # Avoid division by zero
        
#     return numerator / denominator

# An example for testing:
# TRUE_Ex2_beta_5 = get_true_even_moment(5.0, 2)
# TRUE_Ex4_beta_5 = get_true_even_moment(5.0, 4)

# print(TRUE_Ex2_beta_5)
# print(TRUE_Ex4_beta_5)


# Final Project Added
def complex_potential(x):
    """
    Calculates the complex multi-well potential V(x) = (x^2 - 1)(x^2 - 4)(x^2 - 9) / 40
    """
    x2 = x**2
    return (x2 - 1) * (x2 - 4) * (x2 - 9) / 40

def complex_target_density(x, beta):
    """
    Returns the unnormalized target density for the complex potential, p(x; beta) = exp(-beta * V_complex(x))
    """
    return np.exp(-beta * complex_potential(x))

# Ground Truth Values for Complex Potential
TRUE_COMPLEX_Ex = 0.0 # By symmetry
TRUE_COMPLEX_Ex3 = 0.0 # By symmetry

def get_true_even_moment(density_func, beta, power):
    """
    Calculates the exact value for E[x^power] using numerical integration (quadrature).
    This serves as the benchmark/ground truth for the sampling methods.
    
    NOTE: density_func is now passed as an argument (either target_density or complex_target_density)
    """
    if power % 2 != 0:
        return 0.0 # All odd moments are 0
        
    # Integrand for the numerator: x^n * p(x)
    numerator_func = lambda x: (x**power) * density_func(x, beta)
    
    # Integrand for the denominator (Normalization constant Z): p(x)
    denominator_func = lambda x: density_func(x, beta)
    
    # Perform the integration
    # quad(function, lower_limit, upper_limit)
    # quad returns (result, error), we only need the result
    numerator, _ = quad(numerator_func, -np.inf, np.inf)
    denominator, _ = quad(denominator_func, -np.inf, np.inf)
    
    if denominator == 0:
        return np.nan # Avoid division by zero
        
    return numerator / denominator