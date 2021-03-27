from iminuit import util, experimental


def test_expanded():
    def f(x, y, z):
        return x + y + z

    def g(x, a, b):
        return x + a + b

    f2, g2 = experimental.expanded(f, g)

    assert f(1, 2, 3) + g(1, 4, 5) == f2(1, 2, 3, 4, 5) + g2(1, 2, 3, 4, 5)
    assert util.describe(f2) == ["x", "y", "z", "a", "b"]
    assert util.describe(g2) == ["x", "y", "z", "a", "b"]
