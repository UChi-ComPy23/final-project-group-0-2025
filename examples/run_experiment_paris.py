import numpy as np
import matplotlib.pyplot as plt
import os

from src import problem
from src import analysis
from src import metropolis

def run_experiment(beta, initial_x, n_steps, step_size=None, use_tuner=False):
    """
    Runs a single Metropolis-Hastings experiment, computes moments (E[x^n]), and generates plots.
    """
    print("\n" + "=" * 60)
    print(f"EXPERIMENT START | Target Beta = {beta:.1f} | Start X = {initial_x}")
    print("=" * 60)
    
    # Tune step size
    if use_tuner:
        tuned_step = metropolis.tune_step_size(
            beta=beta, 
            initial_x=initial_x, 
            target_rate=0.4
        )
        step_size = tuned_step
    elif step_size is None:
        print("Error: Must provide step_size or set use_tuner=True.")
        return

    # Run the M-H sampler
    chain = metropolis.metropolis_hastings(
        n_steps=n_steps,
        beta=beta,
        initial_x=initial_x,
        step_size=step_size,
        burn_in=0.2, # Fixed 20% burn-in
        verbose=True # Show the final printout
    )
    
    # Define Moment Functions (f(x) = x^n)
    f_x1 = lambda x: x
    f_x2 = lambda x: x**2
    f_x3 = lambda x: x**3
    f_x4 = lambda x: x**4
    
    # Get estimates
    est_Ex1 = analysis.compute_expectation(chain, f_x1)
    est_Ex2 = analysis.compute_expectation(chain, f_x2)
    est_Ex3 = analysis.compute_expectation(chain, f_x3)
    est_Ex4 = analysis.compute_expectation(chain, f_x4)
    
    n = len(chain)
    
    # Apply functions to the whole chain
    chain_f1 = f_x1(chain)
    chain_f2 = f_x2(chain)
    chain_f3 = f_x3(chain)
    chain_f4 = f_x4(chain)
    
    # Get the sample variance of the (transformed) chain
    var_f1 = np.var(chain_f1)
    var_f2 = np.var(chain_f2)
    var_f3 = np.var(chain_f3)
    var_f4 = np.var(chain_f4)
    
    # Calculate the naive SEM
    sem_Ex1 = (var_f1 / n)**0.5
    sem_Ex2 = (var_f2 / n)**0.5
    sem_Ex3 = (var_f3 / n)**0.5
    sem_Ex4 = (var_f4 / n)**0.5
    
    # Get true values
    true_Ex1 = problem.TRUE_Ex
    true_Ex2 = problem.get_true_even_moment(beta, 2)
    true_Ex3 = problem.TRUE_Ex3
    true_Ex4 = problem.get_true_even_moment(beta, 4)
    
    # Print a results table
    print("Moment | Estimate | Naive SEM (±) | True Value | Result")
    print("------ | -------- | ------------- | ---------- | ------")
    print(f"E[x]   | {est_Ex1:>8.4f} | {sem_Ex1:>13.6f} | {true_Ex1:>10.4f} | {'WRONG (Stuck!)' if not np.allclose(est_Ex1, true_Ex1, atol=0.2) else 'Correct'}")
    print(f"E[x^2] | {est_Ex2:>8.4f} | {sem_Ex2:>13.6f} | {true_Ex2:>10.4f} | {'Correct' if np.allclose(est_Ex2, true_Ex2, atol=0.2) else 'WRONG'}")
    print(f"E[x^3] | {est_Ex3:>8.4f} | {sem_Ex3:>13.6f} | {true_Ex3:>10.4f} | {'WRONG (Stuck!)' if not np.allclose(est_Ex3, true_Ex3, atol=0.2) else 'Correct'}")
    print(f"E[x^4] | {est_Ex4:>8.4f} | {sem_Ex4:>13.6f} | {true_Ex4:>10.4f} | {'Correct' if np.allclose(est_Ex4, true_Ex4, atol=0.2) else 'WRONG'}")
    
    # Plot the results using functions in analysis.py
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: The trace of the chain
    analysis.plot_chain_trace(ax1, chain, f"M-H Chain Trace (beta={beta})")
    
    # Plot 2: The histogram
    analysis.plot_histogram(ax2, chain, f"M-H Histogram (beta={beta})", beta)
    
    fig.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"experiment_beta_{beta}.png"
    save_path = os.path.join(script_dir, filename)
    plt.savefig(save_path)
    print(f"\n Saved plot to examples/{filename}")


if __name__ == "__main__":
    init_path = os.path.join("examples", "__init__.py")
    if not os.path.exists(init_path):
        try:
            with open(init_path, "w") as f:
                pass # Create empty file
        except IOError:
            print("Warning: Could not create examples/__init__.py")

    N_SAMPLES = 50000
    
    # Experiment 1: The "Stuck" Chain
    # Hihg beta, force to fail
    run_experiment(
        beta=20.0, 
        initial_x=1.0, 
        n_steps=N_SAMPLES, 
        step_size=0.1,  # Force a small step
        use_tuner=False
    )
    
    # Experiment 2: The "Working" Chain
    # Low beta, start at 0, and auto-tune the step size
    run_experiment(
        beta=2.0, 
        initial_x=0.0, 
        n_steps=N_SAMPLES, 
        use_tuner=True
    )

"""
to run in terminal: python -m examples.run_experiment_paris
"""