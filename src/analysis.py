import numpy as np
from . import problem
import matplotlib.pyplot as plt
from scipy.integrate import quad

def compute_expectation(samples, function_f, weights=None):
    """
    Computes the expected value of a function_f(x) given samples.
    """
    samples = np.array(samples)
    
    # Apply the function to the samples
    f_x = function_f(samples)
    
    if weights is None:
        # Standard Monte Carlo (MCMC, Naive)
        return np.mean(f_x)
    else:
        # Weighted mean for Importance Sampling
        if len(samples) != len(weights):
            raise ValueError("Samples and weights must have the same length.")
        
        weights = np.array(weights)
        
        # Compute the normalized weighted sum
        normalized_weights = weights / np.sum(weights)
        return np.sum(normalized_weights * f_x)

def calculate_metrics(all_runs_estimates, true_value):
    """
    Measures the bias and variance of an estimator across multiple independent runs.
    """
    estimates = np.array(all_runs_estimates)
    mean_estimate = np.mean(estimates)
    
    # Bias = |mean_estimate - true_value|
    bias = np.abs(mean_estimate - true_value)
    
    # Variance of the estimator itself
    variance = np.var(estimates, ddof=1) # Use ddof=1 for sample variance
    
    return {'bias': bias, 'variance': variance}

def plot_chain_trace(ax, chain, title="Chain Trace"):
    """
    Plots the time-series trace of the MCMC chain on a given axis.
    """
    ax.plot(chain, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Step Number")
    ax.set_ylabel("Value (x)")

def plot_histogram(ax, chain, title="Histogram", beta=None):
    """
    Plots a histogram of the chain samples and overlays the true target density.
    """
    # Plot the histogram
    ax.hist(chain, bins=100, density=True, alpha=0.7, label="MCMC Samples")
    
    # Plot the true target density if beta is given
    if beta is not None:
        # Define x range for smooth density plot
        x_true = np.linspace(chain.min() - 0.1, chain.max() + 0.1, 200)
        true_p = problem.target_density(x_true, beta)
        
        Z, _ = quad(lambda x: problem.target_density(x, beta), -np.inf, np.inf)
        
        ax.plot(x_true, true_p / Z, 'r-', linewidth=2, label=f"True Density (beta={beta})")
        ax.legend()
        
    ax.set_title(title)
    ax.set_xlabel("Value (x)")
    ax.set_ylabel("Density")