from RTMinuit._libRTMinuit import *
from math import exp
def f(x,y):
    return exp(y)*(x-2)**2+(y-3)**2

m = Minuit(f)
m.migrad(20)
m.migrad(20)
m.migrad(20)
m.migrad(20)
m.hesse()
m.minos()