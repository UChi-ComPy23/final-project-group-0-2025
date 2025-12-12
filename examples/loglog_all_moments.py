import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from src import metropolis
from src import gibbs 
from src import analysis
from src import problem 

# Global Configuration
N_RUNS = 1 # Number of times to repeat MCMC for each sample size (Keep low for speed)
MAX_N_SAMPLES = 10**6
BURN_IN_FRAC = 0.2
TRUE_EX = 0.0 # True mean for E[x], E[x^3]

# Define the sample sizes from 10^2 to 10^6
N_STEPS = np.logspace(2, np.log10(MAX_N_SAMPLES), num=30, dtype=int)
N_STEPS = N_STEPS[N_STEPS > 0] 

# List of moments to analyze: (power, label_suffix)
MOMENT_SPECS = [
    (1, "E[x]"),
    (2, "E[x^2]"),
    (3, "E[x^3]"),
    (4, "E[x^4]")
]

# Core Analysis Function (Universal for MH and Gibbs)

def run_all_moments_error_analysis(D, density_func_target, beta, initial_x, step_size_or_tuner, label_prefix, is_gibbs=False):
    """
    Runs the MCMC chain (MH or Gibbs) and computes the absolute error for all moments (E[x^n]) 
    across the defined range of sample sizes (N_STEPS).

    Returns:
        list of tuples: [(N_steps, errors_moment_1, label_moment_1, step_size), ...]
    """
    
    # 1. Setup (Tuning)
    # Define the density function wrapper needed by the universal MH sampler (must be a function of x only)
    if is_gibbs:
        density_func_for_mh = lambda x: problem.target_density(x, beta)
    else:
        density_func_for_mh = lambda x: density_func_target(x, beta)

    if step_size_or_tuner == 'tune':
        print(f"[{label_prefix}] Tuning step size")
        step_size = metropolis.tune_step_size(initial_x, density_func_for_mh, target_rate=0.4, n_tune_steps=5000)
    else:
        step_size = step_size_or_tuner
        
    initial_x_vector = np.full(D, initial_x) if is_gibbs else initial_x
    
    # 2. Setup Results Structure
    global N_STEPS
    N_steps_to_run = N_STEPS
    all_moment_errors = {spec[0]: [] for spec in MOMENT_SPECS}
    
    # 3. Simulation Loop
    print(f"[{label_prefix}] Running {len(N_steps_to_run)} simulations")
    
    for n in N_steps_to_run:
        # Run the MCMC chain once for this N
        if is_gibbs:
            # Gibbs calls problem.target_density(x, beta) internally
            chain_D = gibbs.gibbs_sampler(D, n, beta, initial_x_vector, step_size, BURN_IN_FRAC, verbose=False)
            chain = chain_D[:, 0] # Use the first component for analysis
        else:
            # MH uses the single-argument density_func_for_mh defined above
            chain = metropolis.metropolis_hastings(n, initial_x, step_size, density_func_for_mh, BURN_IN_FRAC, verbose=False)

        # Compute error for all moments (E[x] through E[x^4]) from the same chain
        for power, _, in MOMENT_SPECS:
            f_x = lambda x: x**power
            
            # Get True Value
            true_Ex_n = problem.get_true_even_moment(density_func_target, beta, power)
            if power % 2 != 0:
                 true_Ex_n = 0.0
            
            # Get Estimate
            Ex_estimate = analysis.compute_expectation(chain, f_x)
            
            # Calculate Error
            error = np.abs(Ex_estimate - true_Ex_n)
            all_moment_errors[power].append(error)

    # 4. Format Results for Plotting
    final_results = []
    for power, suffix in MOMENT_SPECS:
        label = f"{label_prefix} {suffix}"
        final_results.append((N_steps_to_run, np.array(all_moment_errors[power]), label, step_size))
            
    return final_results        

def plot_all_moments_separately(all_results):
    """
    Creates four separate log-log plots, one for each moment (E[x] to E[x^4]), 
    with the C*n^(-1/2) line correctly scaled for each moment.
    """
    
    # Define color map for the 6 experiments
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'] 
    
    global MOMENT_SPECS
    
    # Group results by the moment ( all E[x] lines together)
    results_by_moment = {spec[0]: [] for spec in MOMENT_SPECS}
    
    # Populate results_by_moment structure
    for i in range(len(all_results)):
        experiment_index = i // len(MOMENT_SPECS) 
        moment_power = MOMENT_SPECS[i % len(MOMENT_SPECS)][0]
        results_by_moment[moment_power].append((all_results[i], experiment_index))


    # Iterate and Plot Each Moment
    for moment_power, moment_label in MOMENT_SPECS:
        
        fig, ax = plt.subplots(figsize=(10, 6))
        moment_results_list = results_by_moment[moment_power]
        
        # 1. Calculate and Plot Theoretical Rate (Scaled for this MOMENT)
        
        # Get the working chain data for this specific moment (always the first element in the list)
        working_result = moment_results_list[0][0]
        working_errors = working_result[1] 
        working_N = working_result[0]
        
        # Use a stable point (e.g., the 5th data point, N ~ 100) to estimate C
        C_estimate_point = 5 
        C_theory = working_errors[C_estimate_point] * working_N[C_estimate_point]**(0.5) 
        
        max_N = working_N[-1]
        N_theory = np.logspace(2, np.log10(max_N), 100)
        theoretical_error = C_theory * N_theory**(-0.5)
        
        ax.loglog(N_theory, theoretical_error, 'k--', label=r'$C \cdot n^{-1/2}$ (Theoretical Rate)', zorder=1)

        
        # 2. Plot All Lines for This Single Moment
        for i, (result_tuple, experiment_index) in enumerate(moment_results_list):
            N_steps, errors, label, step_size = result_tuple
            color = colors[experiment_index]

            ax.loglog(N_steps, errors, 
                      marker='.', 
                      linestyle='-', 
                      color=color, 
                      alpha=0.75, 
                      zorder=2,
                      label=label) 

        # 3. Finalize Plot
        ax.set_title(f"Absolute Error Decay for {moment_label} vs. Sample Size (n)", fontsize=14)
        ax.set_xlabel("Number of Samples (n)", fontsize=12)
        ax.set_ylabel(f"Absolute Error |{moment_label} Est - True|", fontsize=12) 
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
        ax.legend(loc='upper right', fontsize=8, title="Experiment (Case)")
        ax.set_ylim(1e-6, 10) 
        
        plt.tight_layout()
        filename = f"convergence_moment_{moment_power}.png"
        plt.savefig(filename)
        print(f"\n Convergence plot saved to {filename}")

if __name__ == "__main__":
    all_results = []
    print("Starting Full Convergence Analysis (All 4 Moments)")
    
    # Case 1: M-H Simple, Working (beta=1.0)
    results = run_all_moments_error_analysis(D=1, density_func_target=problem.target_density, beta=1.0, initial_x=0.0, step_size_or_tuner='tune', label_prefix="M-H (Simple, β=1.0)", is_gibbs=False)
    all_results.extend(results)

    # Case 2: M-H Simple, Stuck (beta=20.0)
    results = run_all_moments_error_analysis(D=1, density_func_target=problem.target_density, beta=20.0, initial_x=1.0, step_size_or_tuner=0.1, label_prefix="M-H (Simple, β=20.0, Stuck)", is_gibbs=False)
    all_results.extend(results)

    # Case 3: M-H Complex, Working (beta=1.0)
    results = run_all_moments_error_analysis(D=1, density_func_target=problem.complex_target_density, beta=1.0, initial_x=0.0, step_size_or_tuner='tune', label_prefix="M-H (Complex, β=1.0)", is_gibbs=False)
    all_results.extend(results)

    # Case 4: M-H Complex, Stuck (beta=20.0)
    results = run_all_moments_error_analysis(D=1, density_func_target=problem.complex_target_density, beta=20.0, initial_x=3.0, step_size_or_tuner=0.05, label_prefix="M-H (Complex, β=20.0, Stuck)", is_gibbs=False)
    all_results.extend(results)

    # Case 5: Gibbs Simple, Working (beta=1.0, D=10)
    D_GIBBS_WORKING = 10
    results = run_all_moments_error_analysis(D=D_GIBBS_WORKING, density_func_target=problem.target_density, beta=1.0, initial_x=0.0, step_size_or_tuner='tune', label_prefix=f"Gibbs (D={D_GIBBS_WORKING}, β=1.0)", is_gibbs=True)
    all_results.extend(results)

    # Case 6: Gibbs Simple, Stuck (beta=20.0, D=20)
    D_GIBBS_STUCK = 20
    results = run_all_moments_error_analysis(D=D_GIBBS_STUCK, density_func_target=problem.target_density, beta=20.0, initial_x=1.0, step_size_or_tuner=0.1, label_prefix=f"Gibbs (D={D_GIBBS_STUCK}, β=20.0, Stuck)", is_gibbs=True)
    all_results.extend(results)

    # Plot Final Results
    plot_all_moments_separately(all_results)