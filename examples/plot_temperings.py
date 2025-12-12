import numpy as np
import scipy as sp
import sys
import os

current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.parallel_tempering import *
from src.simulated_tempering import simulated_tempering
import matplotlib.pyplot as plt
from scipy.integrate import quad

# initialization
f = lambda x: x # mean 
f2 = lambda x: x**2 # actually (X-mu)^2 but mu = 0, variance
f3 = lambda x: x**3 # skewness

# True Value for second moment
true_moment2_V1_beta1 = (quad(lambda x: f2(x)*np.exp(log_p(x,1,V1)),-np.inf, np.inf)[0])/(quad(lambda x: np.exp(log_p(x,1,V1)),-np.inf, np.inf)[0])
true_moment2_V1_beta20 = (quad(lambda x: f2(x)*np.exp(log_p(x,20,V1)),-np.inf, np.inf)[0])/(quad(lambda x: np.exp(log_p(x,20,V1)),-np.inf, np.inf)[0])
true_moment2_V2_beta1 = (quad(lambda x: f2(x)*np.exp(log_p(x,1,V2)),-np.inf, np.inf)[0])/(quad(lambda x: np.exp(log_p(x,1,V2)),-np.inf, np.inf)[0])
true_moment2_V2_beta20 = (quad(lambda x: f2(x)*np.exp(log_p(x,20,V2)),-np.inf, np.inf)[0])/(quad(lambda x: np.exp(log_p(x,20,V2)),-np.inf, np.inf)[0])

Ns = range(1000,12001,200)

# PT
PT_moment1_V1_beta1 = []
PT_moment2_V1_beta1 = []
PT_moment3_V1_beta1 = []
for N in Ns:
    t_PT_moment1_V1_beta1 = []
    t_PT_moment2_V1_beta1 = []
    t_PT_moment3_V1_beta1 = []

    for i in range(10):
        betas = generate_betas(10, beta_min=0.1, beta_max=1.0)
        xs = parallel_tempering(N,N//5,betas,3.0,V1,0.5)
        t_PT_moment1_V1_beta1.append(np.mean(f(xs)))
        t_PT_moment2_V1_beta1.append(np.mean(f2(xs)))
        t_PT_moment3_V1_beta1.append(np.mean(f3(xs)))
    
    PT_moment1_V1_beta1.append(np.mean(t_PT_moment1_V1_beta1))
    PT_moment2_V1_beta1.append(np.mean(t_PT_moment2_V1_beta1))
    PT_moment3_V1_beta1.append(np.mean(t_PT_moment3_V1_beta1))

PT_moment1_V1_beta20 = []
PT_moment2_V1_beta20 = []
PT_moment3_V1_beta20 = []
for N in Ns:
    t_PT_moment1_V1_beta20 = []
    t_PT_moment2_V1_beta20 = []
    t_PT_moment3_V1_beta20 = []

    for i in range(10):
        betas = generate_betas(20, beta_min=0.01, beta_max=20.0)
        xs = parallel_tempering(N,N//5,betas,3,V1,0.7)
        t_PT_moment1_V1_beta20.append(np.mean(f(xs)))
        t_PT_moment2_V1_beta20.append(np.mean(f2(xs)))
        t_PT_moment3_V1_beta20.append(np.mean(f3(xs)))

    PT_moment1_V1_beta20.append(np.mean(t_PT_moment1_V1_beta20))
    PT_moment2_V1_beta20.append(np.mean(t_PT_moment2_V1_beta20))
    PT_moment3_V1_beta20.append(np.mean(t_PT_moment3_V1_beta20))

PT_moment1_V2_beta1 = []
PT_moment2_V2_beta1 = []
PT_moment3_V2_beta1 = []
for N in Ns:
    t_PT_moment1_V2_beta1 = []
    t_PT_moment2_V2_beta1 = []
    t_PT_moment3_V2_beta1 = []

    for i in range(10):
        betas = generate_betas(10, beta_min=0.1, beta_max=1.0)
        xs = parallel_tempering(N,N//5,betas,4.0,V2,0.5)
        t_PT_moment1_V2_beta1.append(np.mean(f(xs)))
        t_PT_moment2_V2_beta1.append(np.mean(f2(xs)))
        t_PT_moment3_V2_beta1.append(np.mean(f3(xs)))
    
    PT_moment1_V2_beta1.append(np.mean(t_PT_moment1_V2_beta1))
    PT_moment2_V2_beta1.append(np.mean(t_PT_moment2_V2_beta1))
    PT_moment3_V2_beta1.append(np.mean(t_PT_moment3_V2_beta1))   

PT_moment1_V2_beta20 = []
PT_moment2_V2_beta20 = []
PT_moment3_V2_beta20 = []
for N in Ns:
    t_PT_moment1_V2_beta20 = []
    t_PT_moment2_V2_beta20 = []
    t_PT_moment3_V2_beta20 = []

    for i in range(10):
        betas = generate_betas(20, beta_min=0.01, beta_max=20.0)
        xs = parallel_tempering(N,N//5,betas,4.0,V2,0.7)
        t_PT_moment1_V2_beta20.append(np.mean(f(xs)))
        t_PT_moment2_V2_beta20.append(np.mean(f2(xs)))
        t_PT_moment3_V2_beta20.append(np.mean(f3(xs)))
    PT_moment1_V2_beta20.append(np.mean(t_PT_moment1_V2_beta20))
    PT_moment2_V2_beta20.append(np.mean(t_PT_moment2_V2_beta20))
    PT_moment3_V2_beta20.append(np.mean(t_PT_moment3_V2_beta20))

# ST
ST_moment1_V1_beta1 = []
ST_moment2_V1_beta1 = []
ST_moment3_V1_beta1 = []
for N in Ns:
    t_ST_moment1_V1_beta1 = []
    t_ST_moment2_V1_beta1 = []
    t_ST_moment3_V1_beta1 = []
    
    for i in range(10):
        betas = generate_betas(10, beta_min=0.01, beta_max=1.0)
        xs, beta_idx = simulated_tempering(N,N//5,betas,3,V1,df_std=0.5)
        target_xs = xs[beta_idx == 0]
        t_ST_moment1_V1_beta1.append(np.mean(f(target_xs)))
        t_ST_moment2_V1_beta1.append(np.mean(f2(target_xs)))
        t_ST_moment3_V1_beta1.append(np.mean(f3(target_xs)))
    
    ST_moment1_V1_beta1.append(np.mean(t_ST_moment1_V1_beta1))
    ST_moment2_V1_beta1.append(np.mean(t_ST_moment2_V1_beta1))
    ST_moment3_V1_beta1.append(np.mean(t_ST_moment3_V1_beta1))

ST_moment1_V1_beta20 = []
ST_moment2_V1_beta20 = []
ST_moment3_V1_beta20 = []
for N in Ns:
    t_ST_moment1_V1_beta20 = []
    t_ST_moment2_V1_beta20 = []
    t_ST_moment3_V1_beta20 = []
    for i in range(10):
        betas = generate_betas(15, beta_min=0.1, beta_max=20.0)
        xs, beta_idx = simulated_tempering(N,N//5,betas,3,V1,df_std=0.7)
        target_xs = xs[beta_idx == 0]
        t_ST_moment1_V1_beta20.append(np.mean(f(target_xs)))
        t_ST_moment2_V1_beta20.append(np.mean(f2(target_xs)))
        t_ST_moment3_V1_beta20.append(np.mean(f3(target_xs)))
    ST_moment1_V1_beta20.append(np.mean(t_ST_moment1_V1_beta20))
    ST_moment2_V1_beta20.append(np.mean(t_ST_moment2_V1_beta20))
    ST_moment3_V1_beta20.append(np.mean(t_ST_moment3_V1_beta20))

ST_moment1_V2_beta1 = []
ST_moment2_V2_beta1 = []
ST_moment3_V2_beta1 = []
for N in Ns:
    t_ST_moment1_V2_beta1 = []
    t_ST_moment2_V2_beta1 = []
    t_ST_moment3_V2_beta1 = []
    for i in range(10):
        betas = generate_betas(10, beta_min=0.1, beta_max=1.0)
        xs, beta_idx = simulated_tempering(N,N//5,betas,4,V2,df_std=0.5)
        target_xs = xs[beta_idx == 0]
        t_ST_moment1_V2_beta1.append(np.mean(f(target_xs)))
        t_ST_moment2_V2_beta1.append(np.mean(f2(target_xs)))
        t_ST_moment3_V2_beta1.append(np.mean(f3(target_xs)))
    ST_moment1_V2_beta1.append(np.mean(t_ST_moment1_V2_beta1))
    ST_moment2_V2_beta1.append(np.mean(t_ST_moment2_V2_beta1))
    ST_moment3_V2_beta1.append(np.mean(t_ST_moment3_V2_beta1))

ST_moment1_V2_beta20 = []
ST_moment2_V2_beta20 = []
ST_moment3_V2_beta20 = []
for N in Ns:
    t_ST_moment1_V2_beta20 = []
    t_ST_moment2_V2_beta20 = []
    t_ST_moment3_V2_beta20 = []
    for i in range(10):
        betas = generate_betas(30, beta_min=0.01, beta_max=20.0)
        xs, beta_idx = simulated_tempering(N,N//5,betas,5,V2,df_std=1)
        target_xs = xs[beta_idx == 0]
        t_ST_moment1_V2_beta20.append(np.mean(f(target_xs)))
        t_ST_moment2_V2_beta20.append(np.mean(f2(target_xs)))
        t_ST_moment3_V2_beta20.append(np.mean(f3(target_xs)))
    ST_moment1_V2_beta20.append(np.mean(t_ST_moment1_V2_beta20))
    ST_moment2_V2_beta20.append(np.mean(t_ST_moment2_V2_beta20))
    ST_moment3_V2_beta20.append(np.mean(t_ST_moment3_V2_beta20))

# plot them
# 2*2: rows approximation value, error; columns: first moment, second moment, third moment
# 'color' indicate various V and beta, 'ls' indicates PT or ST
# V1 beta1: red; V1 beta20:blue; V2 beta1: green; V2 beta20: purple

fig,ax = plt.subplots(3,2,sharex=True,sharey=True,figsize=(12,10))
# PT 1st moment
ax[0,0].plot(Ns,PT_moment1_V1_beta1,color='red',label='PT V1 beta=1',lw=2,alpha=0.6)
ax[0,0].plot(Ns,PT_moment1_V1_beta20,color='blue',label='PT V1 beta=20',lw=2,alpha=0.6)
ax[0,0].plot(Ns,PT_moment1_V2_beta1,color='green',label='PT V2 beta=1',lw=2,alpha=0.6)
ax[0,0].plot(Ns,PT_moment1_V2_beta20,color='purple',label='PT V2 beta=20',lw=2,alpha=0.6)
ax[0,0].legend()
ax[0,0].set_ylabel('1st Moment / Mean')
ax[0,0].grid(alpha=0.5)
ax[0,0].set_title('Parallel Tempering')
# ST 1st moment
ax[0,1].plot(Ns,ST_moment1_V1_beta1,color='red',lw=2,label='ST V1 beta=1',alpha=0.6)
ax[0,1].plot(Ns,ST_moment1_V1_beta20,color='blue',lw=2,label='ST V1 beta=20',alpha=0.6)
ax[0,1].plot(Ns,ST_moment1_V2_beta1,color='green',lw=2,label='ST V2 beta=1',alpha=0.6)
ax[0,1].plot(Ns,ST_moment1_V2_beta20,color='purple',lw=2,label='ST V2 beta=20',alpha=0.6)
ax[0,1].legend()
ax[0,1].grid(alpha=0.5)
ax[0,1].set_title('Simulated Tempering')
# true value

# PT 2st moment
ax[1,0].plot(Ns,PT_moment2_V1_beta1,color='red',label='PT V1 beta=1',lw=2,alpha=0.6)
ax[1,0].plot(Ns,PT_moment2_V1_beta20,color='blue',label='PT V1 beta=20',lw=2,alpha=0.6)
ax[1,0].plot(Ns,PT_moment2_V2_beta1,color='green',label='PT V2 beta=1',lw=2,alpha=0.6)
ax[1,0].plot(Ns,PT_moment2_V2_beta20,color='purple',label='PT V2 beta=20',lw=2,alpha=0.6)
ax[1,0].legend()
ax[1,0].set_ylabel('2nd Moment / Variance')
ax[1,0].grid(alpha=0.5)
# ST 2st moment
ax[1,1].plot(Ns,ST_moment2_V1_beta1,color='red',label='ST V1 beta=1',lw=2,alpha=0.6)
ax[1,1].plot(Ns,ST_moment2_V1_beta20,color='blue',label='ST V1 beta=20',lw=2,alpha=0.6)
ax[1,1].plot(Ns,ST_moment2_V2_beta1,color='green',label='ST V2 beta=1',lw=2,alpha=0.6)
ax[1,1].plot(Ns,ST_moment2_V2_beta20,color='purple',label='ST V2 beta=20',lw=2,alpha=0.6)
ax[1,1].legend()
ax[1,1].grid(alpha=0.5)


# PT 3st moment
ax[2,0].plot(Ns,PT_moment3_V1_beta1,color='red',label='PT V1 beta=1',lw=2,alpha=0.6)
ax[2,0].plot(Ns,PT_moment3_V1_beta20,color='blue',label='PT V1 beta=20',lw=2,alpha=0.6)
ax[2,0].plot(Ns,PT_moment3_V2_beta1,color='green',label='PT V2 beta=1',lw=2,alpha=0.6)
ax[2,0].plot(Ns,PT_moment3_V2_beta20,color='purple',label='PT V2 beta=20',alpha=0.6)
ax[2,0].legend()
ax[2,0].set_ylabel('3rd Moment / Skewness')
ax[2,0].grid(alpha=0.5)
ax[2,0].set_xlabel('N')
# # ST 3st moment
ax[2,1].plot(Ns,ST_moment3_V1_beta1,color='red',label='ST V1 beta=1',lw=2,alpha=0.6)
ax[2,1].plot(Ns,ST_moment3_V1_beta20,color='blue',label='ST V1 beta=20',lw=2,alpha=0.6)
ax[2,1].plot(Ns,ST_moment3_V2_beta1,color='green',label='ST V2 beta=1',lw=2,alpha=0.6)
ax[2,1].plot(Ns,ST_moment3_V2_beta20,color='purple',label='ST V2 beta=20',lw=2,alpha=0.6)
ax[2,1].legend()
ax[2,1].grid(alpha=0.5)
ax[2,1].set_xlabel('N')
fig.suptitle('Approximation of 1st, 2nd, 3rd Moment', size=23)
plt.tight_layout()
plt.savefig('all_temperings.png')
plt.close()

# mse error plot 
def calculate_error(xs,truth):
    return np.sqrt((np.array(xs)-truth)**2)

# error
fig,ax = plt.subplots(3,2,sharex=True,figsize=(12,10))
# PT 1st moment
ax[0,0].semilogy(Ns,calculate_error(PT_moment1_V1_beta1,0),color='red',label='PT V1 beta=1',lw=2,alpha=0.6)
ax[0,0].semilogy(Ns,calculate_error(PT_moment1_V1_beta20,0),color='blue',label='PT V1 beta=20',lw=2,alpha=0.6)
ax[0,0].semilogy(Ns,calculate_error(PT_moment1_V2_beta1,0),color='green',label='PT V2 beta=1',lw=2,alpha=0.6)
ax[0,0].semilogy(Ns,calculate_error(PT_moment1_V2_beta20,0),color='purple',label='PT V2 beta=20',lw=2,alpha=0.6)
ax[0,0].legend()
ax[0,0].set_ylabel('1st Moment/Mean')
ax[0,0].grid(alpha=0.5)
ax[0,0].set_title('Parallel Tempering')
# ST 1st moment
ax[0,1].semilogy(Ns,calculate_error(ST_moment1_V1_beta1,0),color='red',lw=2,label='ST V1 beta=1',alpha=0.6)
ax[0,1].semilogy(Ns,calculate_error(ST_moment1_V1_beta20,0),color='blue',lw=2,label='ST V1 beta=20',alpha=0.6)
ax[0,1].semilogy(Ns,calculate_error(ST_moment1_V2_beta1,0),color='green',lw=2,label='ST V2 beta=1',alpha=0.6)
ax[0,1].plot(Ns,calculate_error(ST_moment1_V2_beta20,0),color='purple',lw=2,label='ST V2 beta=20',alpha=0.6)
ax[0,1].legend()
ax[0,1].grid(alpha=0.5)
ax[0,1].set_title('Simulated Tempering')
# true value

# PT 2st moment
ax[1,0].semilogy(Ns,calculate_error(PT_moment2_V1_beta1,true_moment2_V1_beta1),color='red',label='PT V1 beta=1',lw=2,alpha=0.6)
ax[1,0].semilogy(Ns,calculate_error(PT_moment2_V1_beta20,true_moment2_V1_beta20),color='blue',label='PT V1 beta=20',lw=2,alpha=0.6)
ax[1,0].semilogy(Ns,calculate_error(PT_moment2_V2_beta1,true_moment2_V2_beta1),color='green',label='PT V2 beta=1',lw=2,alpha=0.6)
ax[1,0].semilogy(Ns,calculate_error(PT_moment2_V2_beta20,true_moment2_V2_beta20),color='purple',label='PT V2 beta=20',lw=2,alpha=0.6)
ax[1,0].legend()
ax[1,0].set_ylabel('2nd Moment/Variance')
ax[1,0].grid(alpha=0.5)
# ST 2st moment
ax[1,1].semilogy(Ns,calculate_error(ST_moment2_V1_beta1,true_moment2_V1_beta1),color='red',label='ST V1 beta=1',lw=2,alpha=0.6)
ax[1,1].semilogy(Ns,calculate_error(ST_moment2_V1_beta20,true_moment2_V1_beta20),color='blue',label='ST V1 beta=20',lw=2,alpha=0.6)
ax[1,1].semilogy(Ns,calculate_error(ST_moment2_V2_beta1,true_moment2_V2_beta1),color='green',label='ST V2 beta=1',lw=2,alpha=0.6)
ax[1,1].semilogy(Ns,calculate_error(ST_moment2_V2_beta20,true_moment2_V2_beta20),color='purple',label='ST V2 beta=20',lw=2,alpha=0.6)
ax[1,1].legend()
ax[1,1].grid(alpha=0.5)

# PT 3st moment
ax[2,0].semilogy(Ns,calculate_error(PT_moment3_V1_beta1,0),color='red',label='PT V1 beta=1',lw=2,alpha=0.6)
ax[2,0].semilogy(Ns,calculate_error(PT_moment3_V1_beta20,0),color='blue',label='PT V1 beta=20',lw=2,alpha=0.6)
ax[2,0].semilogy(Ns,calculate_error(PT_moment3_V2_beta1,0),color='green',label='PT V2 beta=1',lw=2,alpha=0.6)
ax[2,0].semilogy(Ns,calculate_error(PT_moment3_V2_beta20,0),color='purple',label='PT V2 beta=20',alpha=0.6)
ax[2,0].legend()
ax[2,0].set_ylabel('3rd Moment/Skewness')
ax[2,0].set_xlabel('N')
ax[2,0].grid(alpha=0.5)
# # ST 3st moment
ax[2,1].semilogy(Ns,calculate_error(ST_moment3_V1_beta1,0),color='red',label='ST V1 beta=1',lw=2,alpha=0.6)
ax[2,1].semilogy(Ns,calculate_error(ST_moment3_V1_beta20,0),color='blue',label='ST V1 beta=20',lw=2,alpha=0.6)
ax[2,1].semilogy(Ns,calculate_error(ST_moment3_V2_beta1,0),color='green',label='ST V2 beta=1',lw=2,alpha=0.6)
ax[2,1].semilogy(Ns,calculate_error(ST_moment3_V2_beta20,0),color='purple',label='ST V2 beta=20',lw=2,alpha=0.6)
fig.suptitle('Square Error of 1st, 2nd, 3rd Moment', size=21)
ax[2,1].legend()
ax[2,1].grid(alpha=0.5)
ax[2,1].set_xlabel('N')
plt.tight_layout()
plt.savefig('temperings_errors.png')
plt.cloes()