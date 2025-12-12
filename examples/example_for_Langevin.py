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

ula = UnadjustedLangevin(
    gradV = lambda x: gradV_tri(x, beta=1.0),
    step_size=0.01,
    pi0_sampler=pi0_sampler,
    h=h,
)


def main():
    # reference value I_ref \approx E[h(X)]
    N_ref = 200000
    ref_samples = ula.sample_chain(N_ref)
    I_ref = np.mean(h(ref_samples))
    print("Reference value I_ref ≈", I_ref)

    # log–log
    Ns = np.logspace(2, 5, 15, dtype=int)   # 1e2 ~ 1e5
    M = 10  # taking average to avoid variance

    errors = []

    for N in Ns:
        ests = []
        for _ in range(M):
            samples = ula.sample_chain(N)
            ests.append(np.mean(h(samples)))
        ests = np.array(ests)
        rmse = np.sqrt(np.mean((ests - I_ref)**2))
        errors.append(rmse)
        print(f"N={N:6d}, RMSE={rmse:.4e}")

    errors = np.array(errors)

    # benchmark line C * N^{-1/2}
    C = errors[0] * np.sqrt(Ns[0])
    ref_line = C / np.sqrt(Ns)

    plt.figure(figsize=(7, 5))
    plt.loglog(Ns, errors, "o-", label="ULA error (RMSE)")
    plt.loglog(Ns, ref_line, "--", label=r"$C N^{-1/2}$")
    plt.xlabel(r"$N$")
    plt.ylabel("RMSE")
    plt.title("ULA error vs N (log-log)")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()