class Gradient:
    #from http://code.activestate.com/recipes/266466-html-colors-tofrom-rgb-tuples/
    @classmethod
    def color_for(cls, v, min=0., max=1., startcolor=(163, 254, 186),
                  stopcolor=(255, 117, 117)):
        c = [0] * 3
        for i, sc in enumerate(startcolor):
            offset = 1.0 * (v-min) * (stopcolor[i]-startcolor[i]) / (max-min)
            c[i] = round(startcolor[i] + offset)
        return tuple(c)

    @classmethod
    def xcolor_for(cls, v):
        c = cls.color_for(abs(v))
        return '[rgb]{%3.2f,%3.2f,%3.2f}' % (c[0]/255., c[1]/255., c[2]/255.)

    @classmethod
    def rgb_color_for(cls, v):
        c = cls.color_for(abs(v))
        return 'rgb(%d,%d,%d)' % c
