from iminuit import Minuit


def cost(x, y, z):
    return (x - 1) ** 2 + (y - x) ** 2 + (z - 2) ** 2


m = Minuit(cost, print_level=0, pedantic=False)
m.migrad()
m.draw_mncontour("x", "y", nsigma=4)
