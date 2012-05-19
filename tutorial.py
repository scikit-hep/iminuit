# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <codecell>

from RTMinuit import *

# <codecell>

#There are three ways to define a function for minuit
#first like normal function
def f(xabc,y,z):
    return (xabc-1.)**2 + (y-2.)**2 + (z-3.)**2 -1.
m = Minuit(f)#if you don't like verbosity of minuit pass printlevel=-1
#You may want to do these two. 
#m.set_up(0.5)
#m.set_strategy(2)
m.migrad()#look at your terminal for usual minuit printout
print m.matrix_accurate(), m.migrad_ok() #some useful function for checking result
#m.hesse()
print m.args
print m.values
print m.errors
display( m.html_results())
display( m.html_error_matrix())
x = m.html_error_matrix()
print m.list_of_fixed_param()
m.html_error_matrix()._repr_html_()

# <codecell>

#second way is to pass a callable object
#this is useful if your function needs to be computed on data
class F:
    def __init__(self,data):
        self.data = data
    def __call__(self,x,y,z):
        return (x-self.data[0])**2 + (y-self.data[1])**2 + (z-self.data[2])**2 -1.
f = F([1,2,3])
m = Minuit(f)
m.migrad()
print m.values,m.errors

# <codecell>

#third way is a more advanced one if you need to construct function that take variable number of argument
#this is the way dist_fit makes magic generic chi^2 ML function
class Struct:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
class CustomFunction:
    def __init__(self, order):
        self.order=order
        varnames = ['c%d'%i for i in range(order)]
        #now Poly has signature of f(c0,c1,.....,c<order-1>)
        self.func_code = Struct(co_argcount=order,co_varnames=varnames)
        self.func_defaults = None#optional but makes vectorize happy
    def __call__(self,*arg):
        s = 0
        for i in range(self.order):
            s+=(arg[i]-i)**2
        return s
f=CustomFunction(10)
m = Minuit(f)
m.migrad()
print m.values,m.errors
print m.args

# <codecell>

#limiting and fixing parameter example
def f(x,y,z):
    return (x-1.)**2 + (y-2.)**2 + (z-3.)**2 -1.
m = Minuit(f, x=5., fix_z=True,fix_y=True)#make start value
m.migrad()
print m.args
print m.list_of_fixed_param()
m.list_of_vary_param()

# <codecell>

m = Minuit(f, limit_y=(-10,10))#bound y to some range
m.migrad()
print m.args

# <codecell>

m = Minuit(f, error_y=1.)#initial step for y
m.migrad()
m.minos('y')
print m.args

# <codecell>

#getting minos error
mne = m.minos_errors()
print mne
print mne['x'].eplus

# <codecell>

i = 10
'd{i:e}d'.format(**locals())

# <codecell>

a = ['a','b','c']
i = 2
'{a[i]:s}'.format(**locals())

# <codecell>


# <codecell>


# <codecell>


