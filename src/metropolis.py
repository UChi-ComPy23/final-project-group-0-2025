import numpy as np
from . import problem 

def metropolis_hastings(n_steps, beta, initial_x, step_size, burn_in=0.1,
                        return_rate=False, verbose=True):
    """
    Runs the Metropolis-Hastings sampler for the double-well potential.

    Args:
        n_steps (int): Number of samples to return (after burn-in).
        beta (float): Inverse temperature (difficulty knob).
        initial_x (float): Starting position of the chain.
        step_size (float): Standard deviation (sigma) for the symmetric Gaussian proposal.
        burn_in (float): Fraction/number of steps to discard.
        return_rate (bool): If True, returns (chain, acceptance_rate).
        verbose (bool): If True, prints the final acceptance rate.

    Returns:
        np.ndarray or tuple: The chain of samples.
    """
    
    # Validate burn-in
    if burn_in < 1:
        burn_in_steps = int(n_steps * burn_in)
    else:
        burn_in_steps = int(burn_in)
    
    total_steps = n_steps + burn_in_steps
    chain = np.zeros(total_steps)
    chain[0] = initial_x
    
    current_x = initial_x
    current_p = problem.target_density(current_x, beta)
    
    accepted_count = 0
    
    for i in range(1, total_steps):
        # 1. Propose a new state (symmetric proposal): x' = x + N(0, step_size)
        proposed_x = current_x + np.random.normal(0, step_size)
        
        # 2. Calculate target density at new state
        proposed_p = problem.target_density(proposed_x, beta)
        
        # 3. Calculate acceptance probability (alpha)
        # alpha = min(1, p(x') / p(x)) = min(1, p(x_proposed) / p(x_current)) since the proposal q(x'|x) is symmetric.
        
        if current_p == 0: # Avoid division by zero if starting in a bad spot
            alpha = 1.0 # If we start in a zero-prob region, any move is good
        else:
            alpha = min(1.0, proposed_p / current_p)
            
        # 4. Accept or reject the move
        if np.random.rand() < alpha:
            # Accept
            current_x = proposed_x
            current_p = proposed_p
            accepted_count += 1
        # else:
            # Reject, so current_x remains the same
            
        # Record the current state (repeated if rejected)
        chain[i] = current_x
        
    acceptance_rate = accepted_count / total_steps
    
    # Print statement conditional
    if verbose:
        print(f"[M-H] Beta: {beta:.2f}, Step: {step_size:.2f}, Rate: {acceptance_rate:.2%}")
    
    # Return chain after removing burn-in
    final_chain = chain[burn_in_steps:]
    
    # Determine what to return
    if return_rate:
        return final_chain, acceptance_rate
    else:
        return final_chain

def tune_step_size(beta, initial_x, target_rate=0.4, n_tune_steps=1000, 
                   tolerance=0.05, max_iter=20):
    """
    Tunes the M-H step_size to target an acceptance rate of ~40-50%.
    """
    
    print(f"[Tuner] Tuning step size for beta = {beta} (target rate = {target_rate:.0%})...")
    
    step_size = 1.0  # Initial guess
    damping = 0.8    # Helps stabilize the search

    for i in range(max_iter):
        # Run a short, silent M-H test and get the rate
        _, rate = metropolis_hastings(n_tune_steps, beta, initial_x, step_size, 
                                      burn_in=0.2, return_rate=True, verbose=False)
        
        # Check if we are close enough
        error = rate - target_rate
        if abs(error) < tolerance:
            print(f"[Tuner] Success! Final step: {step_size:.4f}, Rate: {rate:.2%}")
            return step_size
        
        # If not, adjust step size based on error (Proportional control)
        # If rate is too high (error > 0), increase step_size to lower the rate.
        # If rate is too low (error < 0), decrease step_size to raise the rate.
        adjustment_factor = 1.0 + (error * damping)
        step_size *= adjustment_factor
        
        # Keep step_size within sensible bounds
        step_size = np.clip(step_size, 0.001, 10.0)

    print(f"[Tuner] Warning: Max iterations reached. Best step: {step_size:.4f}, Rate: {rate:.2%}")
    return step_size