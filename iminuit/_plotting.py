import warnings
def draw_profile(self, vname, x, y, s=None, band=True, text=True):
    from matplotlib import pyplot as plt
    import numpy as np

    x = np.array(x)
    y = np.array(y)
    #print x
    if s is not None:
        s = np.array(s, dtype=bool)
        x = x[s]
        y = y[s]

    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel(vname)
    plt.ylabel('FCN')


    try:
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
        #print x, y
        le = x[minpos] - x[leftpos]
        re = x[rightpos] - x[minpos]
        #print (le, re)

        if band:
            plt.axvspan(x[leftpos], x[rightpos], facecolor='g', alpha=0.5)

        if text:
            plt.figtext(0.5, 0.5,
                '%s = %7.3e ( -%7.3e , +%7.3e)' % (vname, x[minpos], le, re),
                ha='center')
    except ValueError:
        warnings.warn(RuntimeWarning('band and text is requested but'
                                     ' the bound is too narrow.'))

    return x, y, s


def draw_contour(self, x, y, bins=20, bound=2, args=None, show_sigma=False):
#def draw_contour(self, var1, var2, bins=12, bound1=None, bound2=None, lh=True):
    from matplotlib import pyplot as plt
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

def mncontour_grid(self, x, y, numpoints=20, nsigma=2, sigma_res=4, bins=100, edges=False):
    import numpy as np
    from matplotlib import mlab
    dfcn = []
    xps = []
    yps = []
    sigmas = np.linspace(0.1,nsigma+0.5,sigma_res*nsigma)
    for i, this_sig in enumerate(sigmas):
        xminos, yminos, pts = self.mncontour(x, y, numpoints=numpoints,
                                                sigma=this_sig)
        if len(pts)==0:
            warnings.warn(RuntimeWarning('Fail mncontour for %s, %s, sigma=%f'%(x,y,this_sig)))
            continue
        
        xp, yp = zip(*pts)
        xps.append(xp)
        yps.append(yp)
        dfcn.append([this_sig]*len(pts))

    #add the minimum in
    xps.append([self.values[x]])
    yps.append([self.values[y]])
    dfcn.append([0])

    #making grid data x ,y z
    fx, fy, fz = np.hstack(xps), np.hstack(yps), np.hstack(dfcn)

    maxx ,minx = max(fx), min(fx)
    maxy, miny = max(fy), min(fy)

    #extend bound a bit
    maxx += 0.05*(maxx-minx)
    minx -= 0.05*(maxx-minx)
    maxy += 0.05*(maxy-miny)
    miny -= 0.05*(maxy-miny)

    #need constant spacing for linear
    xgrid = np.linspace(minx,maxx, bins)
    ygrid = np.linspace(miny,maxy, bins)
    xstep = (maxx-minx)/(1.0*bins)
    ystep = (maxy-miny)/(1.0*bins)
    #xgrid = np.arange(minx, maxx+xstep/2, xstep) # over cover
    #ygrid = np.arange(miny, maxy+ystep/2, ystep)

    g = mlab.griddata(fx,fy,fz,xgrid,ygrid)

    if edges: #return grid edges instead of mid point (for pcolor)
        xgrid -= xstep/2.
        ygrid -= ystep/2.
        np.resize(xgrid,len(xgrid)+1)
        np.resize(ygrid,len(ygrid)+1)
        xgrid[-1] = xgrid[-2]+xstep/2.
        ygrid[-1] = ygrid[-2]+ystep/2.

    return xgrid, ygrid, g, (xps, yps, dfcn)

def draw_mncontour(self, x, y, bins=100, nsigma=2, numpoints=20, sig_res=4):

    from matplotlib import pyplot as plt

    xgrid, ygrid, g, r = mncontour_grid(self, x, y, numpoints, nsigma, sig_res, bins)
    #g[g.mask] = nsigma+1 #remove the mask

    CS = plt.contour(xgrid, ygrid, g, range(1,nsigma+1))
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel(x)
    plt.ylabel(y)
    return xgrid, ygrid, g, CS
