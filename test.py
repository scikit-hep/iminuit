from RTMinuit._libRTMinuit import *
from math import exp
def f(x,y):
    return exp(y)*(x-2)**2+(y-3)**2

m = Minuit(f)
m.migrad(1000)
m.hesse()
m.minos()
print m.matrix(True,True)
m.print_matrix()