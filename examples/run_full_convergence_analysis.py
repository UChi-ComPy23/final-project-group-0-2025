import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules from src
from src import metropolis
from src import gibbs 
from src import analysis
from src import problem 

# --- Global Configuration ---
N_RUNS = 1 # Number of times to repeat MCMC for each sample size (Increase for smoother data)
MAX_N_SAMPLES = 10**6
N_STEPS_LOGSPACE = 20 # Number of points on the x-axis
BURN_IN_FRAC = 0.2
TRUE_EX = 0.0

def run_error_analysis_mh(density_func, beta, initial_x, step_size_or_tuner=None, label="M-H"):
    """
    Computes the absolute error of E[x] vs. sample size (n) for Metropolis-Hastings.
    """
    density_func_wrapper = lambda x: density_func(x, beta)
    
    if step_size_or_tuner == 'tune':
        print(f"[{label}] Tuning step size...")
        step_size = metropolis.tune_step_size(initial_x, density_func_wrapper, target_rate=0.4, n_tune_steps=5000)
    else:
        step_size = step_size_or_tuner
        
    N_steps = np.logspace(0, np.log10(MAX_N_SAMPLES), num=N_STEPS_LOGSPACE, dtype=int)
    absolute_errors_mean = []
    
    for n in N_steps:
        errors = []
        for _ in range(N_RUNS):
            chain = metropolis.metropolis_hastings(n, initial_x, step_size, 
                                        density_func_wrapper, 
                                        burn_in=BURN_IN_FRAC, verbose=False)
            
            Ex_estimate = analysis.compute_expectation(chain, lambda x: x)
            errors.append(np.abs(Ex_estimate - TRUE_EX))
            
        absolute_errors_mean.append(np.mean(errors))

    return N_steps, np.array(absolute_errors_mean), label, step_size


def run_error_analysis_gibbs(D, beta, initial_x, step_size_or_tuner=None, label="Gibbs"):
    """
    Computes the absolute error of E[x1] vs. sample size (n) for Gibbs Sampler.
    """
    # The Gibbs sampler uses the simple double-well potential target_density
    density_func_1d = lambda x: problem.target_density(x, beta)
    
    if step_size_or_tuner == 'tune':
        print(f"[{label}] Tuning 1D step size...")
        step_size = metropolis.tune_step_size(initial_x, density_func_1d, target_rate=0.4, n_tune_steps=5000)
    else:
        step_size = step_size_or_tuner
        
    initial_x_vector = np.full(D, initial_x)
        
    N_steps = np.logspace(0, np.log10(MAX_N_SAMPLES), num=N_STEPS_LOGSPACE, dtype=int)
    absolute_errors_mean = []
    
    for n in N_steps:
        errors = []
        for _ in range(N_RUNS):
            chain_D = gibbs.gibbs_sampler(
                D=D, n_steps=n, beta=beta, initial_x_vector=initial_x_vector, 
                step_size_1d=step_size, burn_in=BURN_IN_FRAC, verbose=False
            )
            chain_x1 = chain_D[:, 0]
            
            Ex_estimate = analysis.compute_expectation(chain_x1, lambda x: x)
            errors.append(np.abs(Ex_estimate - TRUE_EX))
            
        absolute_errors_mean.append(np.mean(errors))

    return N_steps, np.array(absolute_errors_mean), label, step_size

def plot_loglog(all_results):
    """
    Creates the final log-log plot of Absolute Error vs. N for all results.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    # --- 1. Plot Theoretical Rate ---
    max_N = max(r[0][-1] for r in all_results)
    N_theory = np.logspace(0, np.log10(max_N), 100)
    
    # Calculate C based on the first data point of the first WORKING MCMC line
    N_start, Error_start = all_results[0][0][0], all_results[0][1][0]
    C = Error_start * N_start**(0.5) 
    theoretical_error = C * N_theory**(-0.5)
    ax.loglog(N_theory, theoretical_error, 'k--', label=r'$C \cdot n^{-1/2}$ (Theoretical Rate)', zorder=1)

    # --- 2. Plot All MCMC Error Lines ---
    for i, (N_steps, errors, label, step_size) in enumerate(all_results):
        ax.loglog(N_steps, errors, marker='o', linestyle='-', color=colors[i % len(colors)], alpha=0.8, zorder=2,
                  label=f"{label} (σ={step_size:.3f})")

    # --- 3. Finalize Plot ---
    ax.set_title(r"MCMC Absolute Error for $\mathbb{E}[x]$ vs. Sample Size (n)", fontsize=16)
    ax.set_xlabel("Number of Samples (n)", fontsize=12)
    ax.set_ylabel(r"Absolute Error $|\mathbb{E}[x]|$", fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    plt.savefig("full_convergence_analysis.png")
    print("\n Convergence analysis plot saved to full_convergence_analysis.png")
    

if __name__ == "__main__":
    all_results = []
    print("Starting Full Convergence Analysis (may take a few minutes)...")
    
    # --- Case 1: M-H Simple, Working (beta=1.0) ---
    N, Errors, Label, Step = run_error_analysis_mh(problem.target_density, beta=1.0, initial_x=0.0, step_size_or_tuner='tune', label="M-H (Simple, β=1.0, Working)")
    all_results.append((N, Errors, Label, Step))

    # --- Case 2: M-H Simple, Stuck (beta=20.0) ---
    N, Errors, Label, Step = run_error_analysis_mh(problem.target_density, beta=20.0, initial_x=1.0, step_size_or_tuner=0.1, label="M-H (Simple, β=20.0, Stuck)")
    all_results.append((N, Errors, Label, Step))

    # --- Case 3: M-H Complex, Working (beta=1.0) ---
    N, Errors, Label, Step = run_error_analysis_mh(problem.complex_target_density, beta=1.0, initial_x=0.0, step_size_or_tuner='tune', label="M-H (Complex, β=1.0, Working)")
    all_results.append((N, Errors, Label, Step))

    # --- Case 4: M-H Complex, Stuck (beta=20.0) ---
    N, Errors, Label, Step = run_error_analysis_mh(problem.complex_target_density, beta=20.0, initial_x=3.0, step_size_or_tuner=0.05, label="M-H (Complex, β=20.0, Stuck)")
    all_results.append((N, Errors, Label, Step))

    # --- Case 5: Gibbs Simple, Working (beta=1.0, D=10) ---
    D_GIBBS_WORKING = 10
    N, Errors, Label, Step = run_error_analysis_gibbs(D_GIBBS_WORKING, beta=1.0, initial_x=0.0, step_size_or_tuner='tune', label=f"Gibbs (D={D_GIBBS_WORKING}, β=1.0, Working)")
    all_results.append((N, Errors, Label, Step))

    # --- Case 6: Gibbs Simple, Stuck (beta=20.0, D=20) ---
    D_GIBBS_STUCK = 20
    N, Errors, Label, Step = run_error_analysis_gibbs(D_GIBBS_STUCK, beta=20.0, initial_x=1.0, step_size_or_tuner=0.1, label=f"Gibbs (D={D_GIBBS_STUCK}, β=20.0, Stuck)")
    all_results.append((N, Errors, Label, Step))

    # --- Plot Final Results ---
    plot_loglog(all_results)