import numpy as np
from . import problem
from .metropolis import metropolis_step_1d

def gibbs_sampler(D, n_steps, beta, initial_x_vector, step_size_1d, burn_in=0.1, verbose=True):
    """
    Runs the Gibbs sampler for the D-dimensional separable double-well potential.
    
    The conditional distribution for each component p(x_i | x_{~i}) is separable,
    meaning it is proportional to the 1D double-well target density.
    
    Args:
        D (int): Dimensionality of the problem.
        n_steps (int): Number of total steps (after burn-in).
        beta (float): Inverse temperature.
        initial_x_vector (np.ndarray): Starting position vector (D-dimensional).
        step_size_1d (float): Standard deviation (sigma) for the 1D M-H proposal.
        burn_in (float): Fraction of steps to discard.
        
    Returns:
        np.ndarray: The chain of D-dimensional vectors (N_samples x D).
    """
    if burn_in < 1:
        burn_in_steps = int(n_steps * burn_in)
    else:
        burn_in_steps = int(burn_in)
        
    total_steps = n_steps + burn_in_steps
    chain = np.zeros((total_steps, D))
    
    current_x_vector = np.array(initial_x_vector, dtype=float)
    chain[0, :] = current_x_vector
    
    accepted_count = 0
    
    for t in range(1, total_steps):
        # Start iteration t with the state from t-1
        current_x_vector = np.copy(chain[t-1, :]) 
        
        # Gibbs sweep: Iterate through all D dimensions
        for i in range(D):
            # The conditional distribution p(x_i | x_{~i}) is proportional to the 
            # 1D target density since the potential is separable.
            
            # Perform one 1D M-H step on component x_i
            new_x_i, accepted = metropolis_step_1d(
                current_x=current_x_vector[i],
                step_size=step_size_1d,
                unnormalized_density_func=problem.target_density, # Use the 1D double-well density
                beta=beta
            )
            
            current_x_vector[i] = new_x_i
            if accepted:
                accepted_count += 1
                
        # Record the updated D-dimensional vector
        chain[t, :] = current_x_vector
        
    # The acceptance rate is the total number of accepted single-component moves divided 
    # by the total number of proposed single-component moves (D * total_steps)
    acceptance_rate = accepted_count / (D * total_steps)
    
    if verbose:
        print(f"[Gibbs] D={D}, Beta: {beta:.2f}, Step: {step_size_1d:.4f}, Rate: {acceptance_rate:.2%}")
        
    return chain[burn_in_steps:, :]