import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.scripts_yifu import ClassicalMonteCarlo, ImportanceSampling
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
# ==== Setup classical MC (Normal distribution) ====
true_mean = 0
true_var = 1

sampler = lambda N: np.random.normal(0, 1, N)
h_mean = lambda x: x

# ==== Setup importance sampling (double-well target) ====
beta = 1.0
target_unnorm = lambda x: np.exp(-beta * (x**2 - 1)**2)
proposal_mu, proposal_sigma = 0, 2
proposal_sampler = lambda N: np.random.normal(proposal_mu, proposal_sigma, size=N)
proposal_pdf = lambda x: norm.pdf(x, proposal_mu, proposal_sigma)

# Ns
Ns = np.logspace(2, 5, 20).astype(int)

# Results
mc_mean_vals, mc_var_vals = [], []
is_mean_vals, is_var_vals = [], []

for N in Ns:
    # --- Classical MC ---
    X = sampler(N)
    mc_mean_vals.append(np.mean(X))
    mc_var_vals.append(np.var(X, ddof=1))

    # --- Importance Sampling (mean) ---
    IS_mean = ImportanceSampling(proposal_sampler, proposal_pdf, target_unnorm, h_mean)
    m_est = IS_mean.estimate(N)
    is_mean_vals.append(m_est)

    # --- Importance Sampling (variance) ---
    h_var = lambda x, m=m_est: (x - m)**2
    IS_var = ImportanceSampling(proposal_sampler, proposal_pdf, target_unnorm, h_var)
    is_var_vals.append(IS_var.estimate(N))

# ==== Plot ====
plt.figure(figsize=(12, 10))

# Classical MC mean
plt.subplot(2,2,1)
plt.plot(Ns, mc_mean_vals)
#plt.yscale("log")
plt.axhline(true_mean, linestyle="--")
plt.xscale("log")
plt.title("Classical MC Mean vs N")
plt.xlabel("N")
plt.ylabel("Mean")

# Classical MC variance
plt.subplot(2,2,2)
plt.plot(Ns, mc_var_vals)
#plt.yscale("log")
plt.axhline(true_var, linestyle="--")
plt.xscale("log")
plt.title("Classical MC Variance vs N")
plt.xlabel("N")
plt.ylabel("Variance")

# IS mean
plt.subplot(2,2,3)
plt.plot(Ns, is_mean_vals)
#plt.yscale("log")
plt.xscale("log")
plt.title("Importance Sampling Mean vs N")
plt.xlabel("N")
plt.ylabel("Mean")

# IS variance
plt.subplot(2,2,4)
plt.plot(Ns, is_var_vals)
#plt.yscale("log")
plt.xscale("log")
plt.title("Importance Sampling Variance vs N")
plt.xlabel("N")
plt.ylabel("Variance")

plt.tight_layout()
plt.show()

