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
    The complex potential function V(x) in Double-Well;
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
    hottest temperature for reference) by geometric scaling.
    '''
    if n == 1:
        # if need only one ladder, the target beta should be turned 
        return np.array([beta_max], dtype=np.float64)
    
    # use logspace() to apply geometric scaling.
    # here, we use np.e as a bse
    betas = np.logspace(np.log(beta_min), np.log(beta_max),
                       num=n, base=np.e, dtype=np.float64)
    return betas[::-1]
# ===================================================================

def simulated_Tempering(n_steps, n_burns, betas, b, V, df_std = 1, freq = 1000):
    ''''''
    n_betas = len(betas) # the number of beta
    x_current = np.random.uniform(low=-b, high=b) # initial state X_0
    k_current = 0 # initial index for list 'betas'

    # Apply Pesudo-prior weights to ensures all temperatures ladders 
    # are visited equally often
    # initial them uniformly
    w = np.ones(n_betas)
    # count visited temperature 
    temps_visited = np.zeros(n_betas)

    # initial MC chain X
    X = np.zeros(n_steps, dtype=np.float64)

    # Define log of r(k,l)
    def log_r(k,l):
        '''
        log of Temperature proposal Markov kernel k(r,l):
        r(k,k+1) = r(k,k-1) = 1/2
        r(0,1) = r(n_betas-1, n_betas-2) = 1,
        where r,l is the index of betas.
        '''
        if k == 0:
            return 0 if l == 1 else -np.inf
        elif k == n_betas - 1:
            return 0 if l == n_betas-2 else -np.inf
        else:
            if l == k + 1 or l == k - 1:
                return np.log(1/2)
            else:
                return -np.inf
    
    # define log of f_ST
    def log_fst(x, i):
        '''
        log of joint distribution:
        log(f_st) = log(p(x;beta))+log(w_i),
        where w_i is pesudo prior weight for i-th beta
        '''
        return log_p(x, betas[i], V) + np.log(w[i])
    
    for i in range(n_steps):
        # ----1. "moving" the temperature/beta
        if k_current == 0:
            k = 1 # because r(0,1) = 1
        elif k_current == n_betas - 1:
            k = n_betas - 2 # because r(n_betas-1, n_betas-2) = 1
        else:
            # r(k,k+1) = r(k,k-1) = 1/2
            if np.random.rand() < 0.5:
                k = k_current - 1
            else: 
                k = k_current + 1
        # calculate the acceptance ratio
        log_a = min(0, log_fst(x_current, k) + log_r(k, k_current) - 
                    log_fst(x_current, k_current) - log_r(k_current,k))
        # if accept, update the current k
        if np.log(np.random.rand()) < log_a:
            k_current = k
        
        # track the visited temperature
        temps_visited[k_current] += 1

        # ------ 2. "moving" the state x -------
        std = df_std/np.sqrt(betas[k_current])
        x = x_current + np.random.normal(0, std)
        # calculate the acceptance ratio
        # log(q(z,x)/q(x,z)) = log(1) = 0 because q is symmetric normal distribution
        log_alpha = min(0, log_fst(x, k_current) - log_fst(x_current, k_current))
        # if accept, update the current x
        if np.log(np.random.rand()) < log_alpha:
            x_current = x
        
        # store the x
        X[i] = x_current

        # adjust the pesudo prior weight w
        if i > 0 and i < n_burns and (i+1)%freq == 0:
            avg_visit = temps_visited.mean() # average time of visit for each beta/temperature
            freq_visit = np.where(temps_visited > 1, temps_visited, 1) # find whose visit is 
            # lower than avergae
            w *= avg_visit/freq_visit # scale variously for each temperatrue 
            w /= w.sum() # normalize it
            temps_visited.fill(0) # reset 
    
    return X[n_burns:]

if __name__ == '__main__':
    pass