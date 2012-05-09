import ROOT
from inspect import getargspec
import sys
from array import array
#from common import FCN
class Minuit:
    
    def __init__(self,f,**kwds):
        self.fcn = FCN(f)
        
        args,_,_,_ = getargspec(f)
        narg = len(args)
        
        #maintain 2 dictionary 1 is position to varname
        #and varname to position
        self.varname = args
        self.pos2var = {i:k for i,k in enumerate(args)}
        self.var2pos = {k:i for i,k in enumerate(args)}

        self.tmin = ROOT.TMinuit(narg)
        self.prepare(**kwds)
        
        self.last_migrad_result = 0
        self.args,self.values,self.errors=None,None,None
        
    def prepare(self,**kwds):
        self.tmin.SetFCN(self.fcn)

        for i,varname in self.pos2var.items():
            initialvalue = kwds[varname] if varname in kwds else 0.
            initialstep = kwds['error_'+varname] if 'error_'+varname in kwds else 0.1
            lrange,urange = kwds['limit_'+varname] if 'limit_'+varname in kwds else (0.,0.)
            ierflg = self.tmin.DefineParameter(i,varname,initialvalue,initialstep,lrange,urange)
            assert(ierflg==0)
        #now fix parameter
        for i,varname in self.pos2var.items():
            if 'fix_'+varname in kwds: self.tmin.FixParameter(i)
        
    def set_up(self,up):
        return self.tmin.SetErrorDef(up)
    
    def set_printlevle(self,lvl):
        return self.tmin.SetPrintLevel(lvl)
    
    def migrad(self):
        """user can check if the return status is not 0"""
        #internally PyRoot store 1 FCN globally 
        #so we need to change it to the correct one every time
        #It's limitation of C++
        self.tmin.SetFCN(self.fcn)
        self.last_migrad_result = self.tmin.Migrad()
        self.set_ave()
        return self.last_migrad_result 
    
    def migrad_ok(self):
        return self.last_migrad_result==0

    def hesse(self):
        self.tmin.SetFCN(self.fcn)
        self.tmin.mnhess()
        self.set_ave()
    
    def minos(self):
        self.tmin.SetFCN(self.fcn)
        self.tmin.mnmnos()
        self.set_ave()
    
    def set_ave(self):
        """set args values and errors"""
        
        tmp_values = {}
        tmp_errors = {}
        for i,varname in self.pos2var.items():
            tmp_val = ROOT.Double(0.)
            tmp_err = ROOT.Double(0.)
            self.tmin.GetParameter(i,tmp_val,tmp_err)
            tmp_values[varname] = float(tmp_val)
            tmp_errors[varname] = float(tmp_err)
        self.values = tmp_values
        self.errors = tmp_errors
        
        val = self.values
        tmparg = []
        for arg in self.varname:
            tmparg.append(val[arg])
        self.args = tuple(tmparg)

    def mnstat(self):
        fmin = ROOT.Double(0.)
        fedm = ROOT.Double(0.)
        errdef = ROOT.Double(0.)
        #void mnstat(Double_t& fmin, Double_t& fedm, Double_t& errdef, Int_t& npari, Int_t& nparx, Int_t& istat)

class FCN:
    #cdef f
    def __init__(self,f):
        self.f = f
        args,_,_,_ = getargspec(f)
        narg = len(args)
        self.narg = narg
    def __call__(self,npar,gin,f,par,flag):
        #FCN(Int_t&npar, Double_t*gin, Double_t&f, Double_t*par, Int_t flag)
        p = tuple((par[i] for i in range(self.narg)))
        #print p
        ret =self.f(*p)
        f[0] = ret

def main():
    def test(x,y):
        return (x-2)**2 + (y-3)**2 + 1.
    m = Minuit(test,x=2.1,y=2.9)
    m.migrad()
    m.minos()
    print m.values
    print m.errors
   
if __name__ == '__main__':
    main()