from iminuit import Minuit


def cost(a, b, c):
    return (
        a**2
        + 0.25 * a**4
        + a * b
        + ((b - 1) / 0.6) ** 2
        - 2.5 * b * c
        + ((c - 2) / 0.5) ** 2
    )


m = Minuit(cost, 1, 1, 1)
m.migrad()

m.draw_mnmatrix(cl=[1, 2, 3])
