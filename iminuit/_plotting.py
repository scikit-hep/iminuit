#profile(self, vname, bins=100, bound=2, args=None, subtract_min=False)
def draw_profile(self, vname, bins=100, bound=2, args=None, subtract_min=False):
    from matplotlib import pyplot as plt
    import numpy as np
    x, y = self.profile(vname, bins, bound, args, subtract_min)
    x = np.array(x)
    y = np.array(y)

    plt.plot(x,y)
    plt.grid(True)
    plt.xlabel(vname)
    plt.ylabel('FCN')

    minpos = np.argmin(y)

    ymin = y[minpos]
    tmpy = y - ymin
    #now scan for minpos to the right until greater than one
    up = self.errordef
    righty = np.power(tmpy[minpos:] - up, 2)
    #print righty
    right_min = np.argmin(righty)
    rightpos = right_min + minpos
    lefty = np.power((tmpy[:minpos] - up), 2)
    left_min = np.argmin(lefty)
    leftpos = left_min
    #print leftpos, rightpos
    le = x[minpos] - x[leftpos]
    re = x[rightpos] - x[minpos]
    #print (le, re)
    p = plt.axvspan(x[leftpos], x[rightpos], facecolor='g', alpha=0.5)
    plt.figtext(0.5, 0.5, '%s = %7.3e ( -%7.3e , +%7.3e)' % (vname, x[minpos], le, re), ha='center')
    return x,y

def draw_contour(self, x, y, bins=20, bound=2, args=None, show_sigma=False):
#def draw_contour(self, var1, var2, bins=12, bound1=None, bound2=None, lh=True):
    from matplotlib import pyplot as plt
    import numpy as np
    vx, vy, vz = self.contour(x, y, bins, bound, args, subtract_min=True)
    #x1s, x2s, y = val_contour2d(fit, m, var1, var2, bins=bins,
    #                            bound1=bound1, bound2=bound2)
    #y -= np.min(y)

    v = [self.errordef*(i**2) for i in range(1,4)]

    CS = plt.contour(vx, vy, vz, v, colors=['b', 'k', 'r'])
    if not show_sigma:
        plt.clabel(CS, v)
    else:
        tmp = dict((vv,r'%i $\sigma$'%(i+1)) for i,vv in enumerate(v))
        plt.clabel(CS, v, fmt=tmp, fontsize=16)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.axhline(self.values[y], color='k', ls='--')
    plt.axvline(self.values[x], color='k', ls='--')
    plt.grid(True)
    return vx, vy, vz
