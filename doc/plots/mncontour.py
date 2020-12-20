from iminuit import Minuit


def cost(x, y, z):
    return (x - 1) ** 2 + (y - x) ** 2 + (z - 2) ** 2


cost.errordef = Minuit.LEAST_SQUARES

m = Minuit(cost, x=0, y=0, z=0)
m.migrad()
m.draw_mncontour("x", "y")
