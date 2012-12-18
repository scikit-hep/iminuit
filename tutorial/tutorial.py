# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from RTMinuit import Minuit, describe

# <markdowncell>

# ##Really Quick Start
# Let go through a quick course about how to minimize things. If you use PyMinuit before you will find that RTMinuit is very similar to PyMinuit. One notable different is that there is no printMode (we use print_level).

# <rawcell>


# <codecell>

#Let's try to minimize simple (x-1)**2 + (y-2)**2 + (z-3)**2 - 1
#we know easily that the answer has to be
#x=1, y=2, z=3
def f(x,y,z):
    return y**2*(x-1.)**2 + (y-2.)**2 + (z-3.)**2 -1.
describe(f) #RTMinuit magically extract function signature

# <codecell>

m=Minuit(f, x=2, error_x=0.2, limit_x=(-10.,10.), y=10000., fix_y=True, print_level=1)
#The initial value/error are optional but it's nice to do it
#and here is how to use it
#x=2 set intial value of x to 2
#error_x=0.2 set the initial stepsize
#limit_x = (-1,1) set the range for x
#y=2, fix_y=True fix y value to 2
#We do not put any constain on z
#Minuit will warn you about missig initial error/step
#but most of the time you will be fine

# <codecell>

#Boom done!!!!
#you can use m.migrad(print_level=0) to make it quiet
m.migrad();

# <codecell>

m.hesse()

# <codecell>

#and this is how you get the the value
print 'parameters', m.parameters
print 'args', m.args
print 'value', m.values

# <codecell>

#and the error
print 'error', m.errors

# <codecell>

#and function value at the minimum
print 'fval', m.fval

# <codecell>

#covariance, correlation matrix
#remember y is fixed
print 'covariance', m.covariance
print 'matrix()', m.matrix() #covariance
print 'matrix(correlation=True)', m.matrix(correlation=True) #correlation
m.print_matrix() #correlation

# <codecell>

#get mimization status
print m.get_fmin()
print m.get_fmin().is_valid

# <codecell>

#you can run hesse() to get (re)calculate hessian matrix
#What you care is value and Parab(olic) error.
m.hesse()

# <codecell>

#to get minos error you do
m.minos()
print m.get_merrors()['x']
print m.get_merrors()['x'].lower
print m.get_merrors()['x'].upper

# <codecell>

#you can force use print_* to do various html display
m.print_param()

# <headingcell level=1>

# Alternative Ways to define function

# <headingcell level=4>

# Cython

# <codecell>

#sometimes we want speeeeeeed
%load_ext cythonmagic

# <codecell>

%%cython
cimport cython

@cython.binding(True)#you need this otherwise RTMinuit can't extract signature
def cython_f(double x,double y,double z):
    return (x-1.)**2 + (y-2.)**2 + (z-3.)**2 -1.

# <codecell>

#you can always see what RTMinuit will see
print describe(cython_f)

# <codecell>

m = Minuit(cython_f)
m.migrad()
print m.values

# <headingcell level=4>

# Callable object ie: __call__

# <codecell>

x = [1,2,3,4,5]
y = [2,4,6,8,10]# y=2x
class StraightLineChi2:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __call__(self,m,c): #lets try to find slope and intercept
        chi2 = sum((y - m*x+c)**2 for x,y in zip(self.x,self.y))
        return chi2

# <codecell>

chi2 = StraightLineChi2(x,y)
describe(chi2)

# <codecell>

m = Minuit(chi2)
m.migrad()
print m.values

# <headingcell level=4>

# Faking a function signature

# <codecell>

#this is very useful if you want to build a generic cost functor
#this is actually how dist_fit is implemented
x = [1,2,3,4,5]
y = [2,4,6,8,10]# y=2x
class Chi2Functor:
    def __init__(self,f,x,y):
        self.f = f
        self.x = x
        self.y = y
        f_sig = describe(f)
        #this is how you fake function 
        #signature dynamically
        self.func_code = Struct(
                                co_varnames = f_sig[1:], #dock off independent variable
                                co_argcount = len(f_sig)-1
                                )
        self.func_defaults = None #this keeps np.vectorize happy
    def __call__(self,*arg):
        #notice that it accept variable length
        #positional arguments
        chi2 = sum((y-self.f(x,*arg))**2 for x,y in zip(self.x,self.y))
        return chi2

# <codecell>

def linear(x,m,c):
    return m*x+c

def parabola(x,a,b,c):
    return a*x**2 + b*x + c 

# <codecell>

linear_chi2 = Chi2Functor(linear,x,y)
describe(linear_chi2)

# <codecell>

m = Minuit(linear_chi2)
m.migrad();
print m.values

# <codecell>

#now here is the beauty
#you can use the same Chi2Functor now for parabola
parab_chi2 = Chi2Functor(parabola,x,y)
describe(parab_chi2)

# <codecell>

m = Minuit(parab_chi2,x,y)
m.migrad()
print m.values

# <headingcell level=4>

# Last Resort: Forcing function signature

# <codecell>

%%cython
#sometimes you get a function with absolutely no signature
#We didn't put cython.binding(True) here 
def nosig_f(x,y):
    return x**2+(y-4)**2

# <codecell>

#something from swig will give you a function with no
#signature information
try:
    describe(nosig_f)#it raise error
except:
    print 'OH NOOOOOOOO!!!!!'

# <codecell>

#Use forced_parameters
m = Minuit(nosig_f, forced_parameters=('x','y'))

# <codecell>

m.migrad()
print m.values

# <headingcell level=1>

# Console Environment

# <codecell>

#this is just showing off console frontend (you can force it)
from RTMinuit.ConsoleFrontend import ConsoleFrontend
m = Minuit(f, frontend=ConsoleFrontend())

# <codecell>

m.migrad();

# <codecell>


