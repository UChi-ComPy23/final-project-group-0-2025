import numpy as np
# ===================================================================
def V1(x):
    '''
    The easy potential function V(x) in Double-Well;
    V(x) = (x^2 - 1)^2
    '''
    return (x**2 - 1)**2

def V2(x):
    '''
    The complex potential function V(x) in Double-Well
    V(x) = (x^2-1)(x^2-4)(x^2-9)/40
    '''
    return (x**2-1)*(x**2-4)*((x**2-9)/40)
# ===================================================================
def log_p(x, beta, V):
    '''
    log(the target density function p(x,beta) in Double-Well);
    log() makes algorithm numerically stable.
    log(p(x,beta)) = log(e^{-bata*V(x)}) = -beta*V(x)
    '''
    return -beta * V(x)
# ===================================================================
def generate_betas(n, beta_min, beta_max):
    '''
    Generate an array of n betas(Inverse of Temperature) from beta_max(inverse of 
    coldest temperature, for target) to beta_min(inverse of 
    hottest temperature for reference) by logarithmically spacing.
    '''
    if n == 1:
        # if need only one ladder, the target beta should be turned 
        return np.array([beta_max], dtype=np.float64)
    
    # use logspace() to apply logarithmical scaling.
    # base is 10.
    betas = np.logspace(np.log10(beta_min), np.log10(beta_max), num=n)
    return betas[::-1]
# ===================================================================

def parallel_tempering(n_steps, n_burns, betas, b, V, df_std):
    '''
    Performs the Parallel Tempering MCMC, 
    tracking results in a 2D array(matrix).
    finally generate the x_i we want in simulation

    n_steps: number of steps in each chain of beta
    n_burn: number of steps for burn-in, smaller than n_steps
    betas: the arrays of betas(Inverse of Temperature)
    b: boundary [-b,b] for initial state x
    V: potential function V(x)
    df_std: the default standard deviation
    '''
    
    # n_chains, the number of chains is actually the length of betas
    n_chains = len(betas)

    if n_steps <= n_burns:
        raise ValueError('n_burns should be smaller than n_steps!')
    # set a initial state x for each beta by uniform distribution.
    states = np.zeros(n_chains)

    # initialize a 2D-array to store the chain of each beta
    # each row represents a chain of a beta, 
    # each column represents a state of each chain
    chains = np.zeros((n_chains, n_steps))

    for t in range(n_steps):
        # 1. --- M-H step --------
        for i in range(n_chains):
            # At each state t, we do the Metropolis-Hasting algorithm for each chain
            x_current = states[i]
            beta = betas[i]
            # Adaptive scaling for std
            std = df_std/np.sqrt(beta)
            # generate new proposal state x
            x = x_current + np.random.normal(0, std)
            # calculate the ratio to accept the new state x
            # criterion is 
            # e^[-beta*V(x)]/e^[-beta*V(x_i)] = e^[-beta*(V(x)-V(x_i))]
            accept_ratio = log_p(x,beta,V) - log_p(x_current,beta,V)

            if np.log(np.random.rand()) < min(0.0, accept_ratio):
                states[i] = x
        
        # 2. --- Swap step --------
        # improve the previous one by 
        # iterate over even-odd pairs, then odd-even pairs for balanced swapping
        for k in range(2):
            for i in range(k, n_chains - 1, 2):
                l, m = i, i + 1  # Always adjacent chains

                beta_l = betas[l]
                beta_m = betas[m]
                x_l = states[l]
                x_m = states[m]
            # swap_ratio = (beta_l-beta_m)*(V(x_l)-V(x_m))
                swap_ratio = (beta_l-beta_m)*(V(x_l)-V(x_m))
                if np.log(np.random.rand()) < min(0.0, swap_ratio):
                    states[l], states[m] = x_m, x_l
        # Appends current states to their respective chains.
        chains[:,t] = states
    
    # we want the coldest chain after burn-in, which is bata_max, which is the last row of
    # the matrix with columns after burn-in.
    return chains[0, n_burns:]
    

if __name__ == '__main__':
    pass
