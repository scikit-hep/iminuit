from iminuit import Minuit


def cost(x, y, z):
    return (x - 1) ** 2 + (y - x) ** 2 + (z - 2) ** 2


m = Minuit(cost, pedantic=False)
m.migrad()
m.draw_mnprofile("y")
