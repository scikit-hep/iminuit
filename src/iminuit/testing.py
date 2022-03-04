"""
Common test functions for optimizers.

Also see: https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""


def rosenbrock(x, y):
    """
    Rosenbrock function. Minimum: f(1, 1) = 0.

    https://en.wikipedia.org/wiki/Rosenbrock_function
    """
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


rosenbrock.errordef = 1


def rosenbrock_grad(x, y):
    """Gradient of Rosenbrock function."""
    return (-400 * x * (-(x**2) + y) + 2 * x - 2, -200 * x**2 + 200 * y)


def ackley(x, y):
    """
    Ackley function. Minimum: f(0, 0) = 0.

    https://en.wikipedia.org/wiki/Ackley_function
    """
    from math import sqrt, exp, cos, pi, e

    term1 = -20 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))
    term2 = -exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y)))
    return term1 + term2 + 20 + e


ackley.errordef = 1


def beale(x, y):
    """
    Beale function. Minimum: f(3, 0.5) = 0.

    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    term1 = 1.5 - x + x * y
    term2 = 2.25 - x + x * y**2
    term3 = 2.625 - x + x * y**3
    return term1 * term1 + term2 * term2 + term3 * term3


beale.errordef = 1


def matyas(x, y):
    """
    Matyas function. Minimum: f(0, 0) = 0.

    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    return 0.26 * (x**2 + y**2) - 0.48 * x * y


matyas.errordef = 1


def sphere_np(x):
    """
    Sphere function for variable number of arguments. Minimum: f(0, ..., 0) = 0.

    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    import numpy as np

    return np.sum(x**2)


sphere_np.errordef = 1
