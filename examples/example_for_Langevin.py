import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import numpy as np
import matplotlib.pyplot as plt
from src.script_for_Langevin import UnadjustedLangevin

def V_bi(x):
    return (x**2 - 1)**2

def gradV_bi(x, beta=1.0):
    return 2*(x**2-1)*2*x*beta

def V_tri(x):
    return ((x**2 - 1) * (x**2 - 4) * (x**2 - 9)) / 40.0

def gradV_tri(x, beta=1.0):
    # d/dx of ((x^2-1)(x^2-4)(x^2-9))/40
    a = x**2 - 1
    b = x**2 - 4
    c = x**2 - 9
    da = 2*x
    db = 2*x
    dc = 2*x
    return beta * (da*b*c + a*db*c + a*b*dc) / 40.0


# initial sampling
def pi0_sampler():
    return np.random.randn()


# test function
def h(x):
    return x

ula_bi_beta1 = UnadjustedLangevin(
    gradV=lambda x: gradV_bi(x, beta=1.0),
    step_size=0.01,
    pi0_sampler=pi0_sampler,
    h=h,
)

ula_bi_beta20 = UnadjustedLangevin(
    gradV=lambda x: gradV_bi(x, beta=20.0),
    step_size=0.01/20.0,
    pi0_sampler=pi0_sampler,
    h=h,
)

ula_tri_beta1 = UnadjustedLangevin(
    gradV=lambda x: gradV_tri(x, beta=1.0),
    step_size=0.01,
    pi0_sampler=pi0_sampler,
    h=h,
)

ula_tri_beta20 = UnadjustedLangevin(
    gradV=lambda x: gradV_tri(x, beta=20.0),
    step_size=0.01/20.0,
    pi0_sampler=pi0_sampler,
    h=h,
)

def compute_rmse_curve(ula, Ns, M, I_ref):
    errors = []
    for N in Ns:
        ests = []
        for _ in range(M):
            samples = ula.sample_chain(N)
            ests.append(np.mean(h(samples)))
        ests = np.array(ests)
        rmse = np.sqrt(np.mean((ests - I_ref) ** 2))
        errors.append(rmse)
    return np.array(errors)

def main():
    N_ref = 200000
    Ns = np.logspace(2, 5, 15, dtype=int)
    M = 10

    # reference values
    I_bi_1 = np.mean(h(ula_bi_beta1.sample_chain(N_ref)))
    I_bi_20 = np.mean(h(ula_bi_beta20.sample_chain(N_ref)))
    I_tri_1 = np.mean(h(ula_tri_beta1.sample_chain(N_ref)))
    I_tri_20 = np.mean(h(ula_tri_beta20.sample_chain(N_ref)))

    # RMSE curves
    err_bi_1 = compute_rmse_curve(ula_bi_beta1, Ns, M, I_bi_1)
    err_bi_20 = compute_rmse_curve(ula_bi_beta20, Ns, M, I_bi_20)
    err_tri_1 = compute_rmse_curve(ula_tri_beta1, Ns, M, I_tri_1)
    err_tri_20 = compute_rmse_curve(ula_tri_beta20, Ns, M, I_tri_20)

    # reference slope
    C = err_bi_1[0] * np.sqrt(Ns[0])
    ref_line = C / np.sqrt(Ns)

    plt.figure(figsize=(7, 5))

    plt.loglog(Ns, err_bi_1, "o-", label=r"Bi-modal, $\beta=1$")
    plt.loglog(Ns, err_bi_20, "o--", label=r"Bi-modal, $\beta=20$")

    plt.loglog(Ns, err_tri_1, "s-", label=r"Tri-modal, $\beta=1$")
    plt.loglog(Ns, err_tri_20, "s--", label=r"Tri-modal, $\beta=20$")

    plt.loglog(Ns, ref_line, "k:", label=r"$C N^{-1/2}$")

    plt.xlabel(r"$N$")
    plt.ylabel("RMSE")
    plt.title(r"ULA error vs $N$ for bi- and tri-modal potentials ($\varepsilon = 0.01/\beta$)")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
