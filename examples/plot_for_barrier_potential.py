import numpy as np
import matplotlib.pyplot as plt

# Potential function
def V(x):
    return (x**2 - 1)**2

x = np.linspace(-2, 2, 400)
betas = [0.5, 1, 2, 5]

plt.figure(figsize=(8,5))
for beta in betas:
    p = np.exp(-beta * V(x))
    plt.plot(x, p, label=f"beta={beta}")

plt.legend()
plt.xlabel("x")
plt.ylabel("unnormalized p(x;β)")
plt.title("Unnormalized densities for different β")
plt.show()
