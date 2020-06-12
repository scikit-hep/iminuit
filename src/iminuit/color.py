from iminuit._deprecated import deprecated

__all__ = ["Gradient"]


class Gradient:
    """Color gradient.
    """

    _steps = None

    def __init__(self, *steps):
        self._steps = steps

    @classmethod
    @deprecated("use Gradient(...)(...)")
    def color_for(
        cls, v, min=0.0, max=1.0, startcolor=(163, 254, 186), stopcolor=(255, 118, 118)
    ):  # pragma: no cover
        bz = (v - min) / (max - min)
        az = 1.0 - bz
        a = startcolor
        b = stopcolor
        return (az * a[0] + bz * b[0], az * a[1] + bz * b[1], az * a[2] + bz * b[2])

    def __call__(self, v):
        st = self._steps
        z = 0.0
        if v < st[0][0]:
            z = 0.0
            i = 0
        elif v >= st[-1][0]:
            z = 1.0
            i = -2
        else:
            i = 0
            for i in range(len(st) - 1):
                if st[i][0] <= v < st[i + 1][0]:
                    break
            z = (v - st[i][0]) / (st[i + 1][0] - st[i][0])
        az = 1.0 - z
        a = st[i]
        b = st[i + 1]
        return (az * a[1] + z * b[1], az * a[2] + z * b[2], az * a[3] + z * b[3])

    def rgb(self, v):
        return "rgb(%.0f,%.0f,%.0f)" % self(v)
