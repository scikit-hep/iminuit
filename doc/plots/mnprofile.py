from iminuit import Minuit


def cost(x, y, z):
    return (x - 1) ** 2 + (y - x) ** 2 + (z - 2) ** 2


m = Minuit(cost, x=0, y=0, z=0)
m.errordef = 1
m.migrad()
m.draw_mnprofile("y")
