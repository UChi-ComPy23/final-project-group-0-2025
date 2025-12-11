import numpy as np
import matplotlib.pyplot as plt
import os

from src import problem
from src import analysis
from src import metropolis
from src import gibbs

def run_gibbs_experiment(D, beta, n_steps, step_size_1d, initial_x_value, use_tuner=True):
    """
    Runs a D-dimensional Gibbs sampling experiment and analyzes the first component (x1).
    """
    print("\n" + "#" * 20 + f" GIBBS (D={D}) EXPERIMENT START " + "#" * 20)
    print(f"Target Beta = {beta:.1f} | Initial X = {initial_x_value}")
    
    # 1. Initialization
    initial_x_vector = np.full(D, initial_x_value)
    
    # 2. Tuning: Since the step is a 1D M-H step, we tune the 1D M-H sampler
    if use_tuner:
        density_func = lambda x: problem.target_density(x, beta)
        print("[Gibbs Tuner] Tuning 1D M-H step for component-wise update")
        tuned_step = metropolis.tune_step_size(
            initial_x=initial_x_value, 
            unnormalized_density_func=density_func, 
            target_rate=0.4
        )
        step_size_1d = tuned_step
    
    # 3. Run the Gibbs Sampler
    chain_D = gibbs.gibbs_sampler(
        D=D,
        n_steps=n_steps,
        beta=beta,
        initial_x_vector=initial_x_vector,
        step_size_1d=step_size_1d,
        verbose=True
    )
    
    # 4. Analysis: Focus on the first component (x1) for comparison
    chain_x1 = chain_D[:, 0]
    n = len(chain_x1)
    
    # Define moment functions
    f_x1 = lambda x: x
    f_x2 = lambda x: x**2
    f_x3 = lambda x: x**3
    f_x4 = lambda x: x**4
    
    # Get estimates and SEM (applied to the x1 component)
    est_Ex1 = analysis.compute_expectation(chain_x1, f_x1)
    est_Ex2 = analysis.compute_expectation(chain_x1, f_x2)
    est_Ex3 = analysis.compute_expectation(chain_x1, f_x3)
    est_Ex4 = analysis.compute_expectation(chain_x1, f_x4)

    var_f1 = np.var(f_x1(chain_x1))
    var_f2 = np.var(f_x2(chain_x1))
    var_f3 = np.var(f_x3(chain_x1))
    var_f4 = np.var(f_x4(chain_x1))

    sem_Ex1 = (var_f1 / n)**0.5
    sem_Ex2 = (var_f2 / n)**0.5
    sem_Ex3 = (var_f3 / n)**0.5
    sem_Ex4 = (var_f4 / n)**0.5

    # We use the original 1D double-well truth for each component
    true_Ex1 = problem.TRUE_Ex
    true_Ex2 = problem.get_true_even_moment(problem.target_density, beta, 2)
    true_Ex3 = problem.TRUE_Ex3
    true_Ex4 = problem.get_true_even_moment(problem.target_density, beta, 4)
    
    # Print results table
    print("Moment | Estimate | Naive SEM (±) | True Value | Result")
    print("------ | -------- | ------------- | ---------- | ------")
    print(f"E[x1]  | {est_Ex1:>8.4f} | {sem_Ex1:>13.6f} | {true_Ex1:>10.4f} | {'WRONG (Stuck!)' if not np.allclose(est_Ex1, true_Ex1, atol=0.2) else 'Correct'}")
    print(f"E[x1^2]| {est_Ex2:>8.4f} | {sem_Ex2:>13.6f} | {true_Ex2:>10.4f} | {'Correct' if np.allclose(est_Ex2, true_Ex2, atol=0.2) else 'WRONG'}")
    print(f"E[x1^3] | {est_Ex3:>8.4f} | {sem_Ex3:>13.6f} | {true_Ex3:>10.4f} | {'WRONG (Stuck!)' if not np.allclose(est_Ex3, true_Ex3, atol=0.2) else 'Correct'}")
    print(f"E[x1^4] | {est_Ex4:>8.4f} | {sem_Ex4:>13.6f} | {true_Ex4:>10.4f} | {'Correct' if np.allclose(est_Ex4, true_Ex4, atol=0.2) else 'WRONG'}")

    # 5. Plotting 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    analysis.plot_chain_trace(ax1, chain_x1, f"Gibbs Chain Trace (D={D}, beta={beta})")
    
    analysis.plot_histogram(ax2, chain_x1, f"Gibbs Histogram (D={D}, beta={beta})", 
                            beta=beta, 
                            unnormalized_density_func=problem.target_density)
    
    fig.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"gibbs_D_{D}_beta_{beta}.png"
    save_path = os.path.join(script_dir, filename)
    plt.savefig(save_path)
    print(f"\n Saved plot to examples/{filename}")


if __name__ == "__main__":
    N_SAMPLES = 50000
    
    # Gibbs Experiment 1: High Beta, Low Dimension (D=5)
    # Show that Gibbs suffers the same mode-sticking as 1D M-H.
    run_gibbs_experiment(
        D=5,
        beta=20.0,
        n_steps=N_SAMPLES,
        step_size_1d=0.1, # Use the fixed, failing step size from 1D M-H
        initial_x_value=1.0,
        use_tuner=False
    )
    
    # Gibbs Experiment 2: High Beta, Higher Dimension (D=20)
    # Show that the dimensionality (D) does not improve mixing for separable Gibbs.
    run_gibbs_experiment(
        D=20,
        beta=20.0,
        n_steps=N_SAMPLES,
        step_size_1d=0.1, 
        initial_x_value=1.0,
        use_tuner=False
    )

    # Gibbs Experiment 3: Low Beta, Working Case (D=10)
    run_gibbs_experiment(
        D=10,
        beta=1.0,
        n_steps=N_SAMPLES,
        step_size_1d=None, 
        initial_x_value=0.0,
        use_tuner=True
    )