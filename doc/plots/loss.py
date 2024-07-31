from matplotlib import pyplot as plt
import numpy as np


def soft_l1(z):
    return 2 * ((1 + z) ** 0.5 - 1)


x = np.linspace(-3, 3)
z = x**2
plt.plot(x, z, label="linear $\\rho(z) = z$")
plt.plot(x, soft_l1(z), label="soft L1-norm $\\rho(z) = 2(\\sqrt{1+z} - 1)$")
plt.xlabel("studentized residual")
plt.ylabel("cost")
plt.legend()
