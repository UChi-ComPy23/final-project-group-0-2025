import numpy as np

def V(x):
    '''
    The potential function V(x) in Double-Well;
    V(x) = (x^2 - 1)^2
    '''
    return (x**2 - 1)**2

def log_p(x, beta):
    '''
    log(the target density function p(x,beta) in Double-Well);
    Here, we use log to make algorithm numerically stable.
    log(p(x,beta)) = log(e^{-bata*V(x)}) = -beta*V(x)
    '''
    return -beta * V(x)


def generate_temp_ladder(n, beta_min, beta_max):
    '''
    Generate an array of n betas(Inverse of Temperature) between beta_min(inverse of 
    hottest temperature for reference) and beta_max(inverse of coldest temper-
    ature, for target) by log-spacing.
    '''
    if n == 1:
        # if need only one ladder, the target beta should be turned 
        return np.array([beta_max], dtype=np.float64)
    
    # use logspace() to apply log spacing.
    # here, we use np.e asa bse
    return np.logspace(np.log(beta_min), np.log(beta_max),
                       num=n, base=np.e, dtype=np.float64)

def parallel_tempering(n_steps, betas, std = 0.5):
    '''
    Performs the Parallel Tempering MCMC, 
    tracking results in a 2D array(matrix).
    finally generate the x_i we want in simulation
    '''
    # betas is the arrays of betas(Inverse of Temperature)
    # n_chains, the number of chains is actually the length of betas
    n_chains = len(betas)
    stds = np.ones(n_chains)*std
    # default std is 0.5 in MH algorithm

    # set a initial state x for each beta
    states = np.random.randn(n_chains)

    # initialize a 2D-array to store the chain of each beta
    # each row represents a chain of a beta, 
    # each column represents a state of each chain
    chains = np.zeros((n_chains, n_steps))
    
    # Loop n_steps
    for t in range(n_steps):
        # for each chain i, perform one M-H step
        for i in range(n_chains):
            # This is why it is called parallel tempering
            # for each state t, we do the Metropolis-Hasting algorithm for each
            # chain
            x_current = states[i]
            beta = betas[i]
            # stds is an attay that actually the std for each chain of beta.
            std = stds[i]
            # generate new_X
            x = x_current + np.random.normal(0, std)
            # calculate the ratio to accept the new state x
            accept_ratio = log_p(x,beta) - log_p(x_current,beta)

            if np.log(np.random.rand()) <= accept_ratio:
                states[i] = x
        
        # swap: Picks two adjacent chains i and j=i+1. 
        # Calculates swap probability. If accepted, swap two state
        for i in range(n_chains-1):
            beta_i = betas[i]
            beta_j = betas[i+1]
            x_i = states[i]
            x_j = states[i+1]

            # because we will compare it with log(uniform(0,1)),
            # so we do not need to worry the sign of swap_ratio
            swap_ratio = (beta_i - beta_j)*(V(x_i) - V(x_j))

            if np.log(np.random.rand()) <= swap_ratio:
                states[i], states[i+1] = x_j, x_i
        
        # Appends current states to their respective chains.
        chains[:,t] = states
    
    # we want the coldest chain, which is bata_max, which is the last row of
    # the matrix
    return chains[-1,:]
    

if __name__ == '__main__':
    pass