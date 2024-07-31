from iminuit import Minuit, cost
import numpy as np
from matplotlib import pyplot as plt


# custom visualization; x, y, model are taken from outer scope
def viz(args):
    plt.plot(x, y, "ok")
    xm = np.linspace(x[0], x[-1], 100)
    plt.plot(xm, model(xm, *args))


def model(x, a, b):
    return a + b * x


x = np.array([1, 2, 3, 4, 5])
y = np.array([1.03, 1.58, 2.03, 2.37, 3.09])
c = cost.LeastSquares(x, y, 0.1, model)
m = Minuit(c, 0.5, 0.5)
m.interactive(viz)
# m.interactive() also works and calls LeastSquares.visualize
