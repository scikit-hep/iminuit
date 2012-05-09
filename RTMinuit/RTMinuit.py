import ROOT
from inspect import getargspec
import sys
from array import array
from util import *
from FCN import FCN

class Minuit:
    
    def __init__(self,f,f_verbose=False,printlevel=0,**kwds):
        self.fcn = FCN(f,verbose=f_verbose)
        
        args = better_arg_spec(f)
        narg = len(args)
        
        #maintain 2 dictionary 1 is position to varname
        #and varname to position
        self.varname = args
        self.pos2var = {i:k for i,k in enumerate(args)}
        self.var2pos = {k:i for i,k in enumerate(args)}

        self.tmin = ROOT.TMinuit(narg)
        self.set_printlevel(printlevel)
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
        """set UP parameter 1 for chi^2 and 0.5 for log likelihood"""
        return self.tmin.SetErrorDef(up)
    
    def set_printlevel(self,lvl):
        """
        set printlevel -1 quiet, 0 normal, 1 verbose
        """
        return self.tmin.SetPrintLevel(lvl)
    
    def set_strategy(self,strategy):
        """
        set strategy
        """
        return self.tmin.Command('SET STR %d'%strategy)
    
    def command(self,cmd):
        """execute a command"""
        return self.tmin.Command(cmd)
    
    def migrad(self):
        """
            run migrad
            user can check if the return status is not 0
        """
        #internally PyRoot store 1 FCN globally 
        #so we need to change it to the correct one every time
        #It's limitation of C++
        self.tmin.SetFCN(self.fcn)
        self.last_migrad_result = self.tmin.Migrad()
        self.set_ave()
        return self.last_migrad_result 
    
    def migrad_ok(self):
        """check whether last migrad call result is OK"""
        return self.last_migrad_result==0

    def hesse(self):
        """run hesse"""
        self.tmin.SetFCN(self.fcn)
        self.tmin.mnhess()
        self.set_ave()
    
    def minos(self):
        """run minos"""
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
        """
        return named tuple of fmin,fedm,errdef,npari,nparx,istat
        """
        fmin = ROOT.Double(0.)
        fedm = ROOT.Double(0.)
        errdef = ROOT.Double(0.)
        npari = ROOT.Long(0.)
        nparx = ROOT.Long(0.)
        istat = ROOT.Long(0.)
        #void mnstat(Double_t& fmin, Double_t& fedm, Double_t& errdef, Int_t& npari, Int_t& nparx, Int_t& istat)
        self.tmin.mnstat(fmin,fedm,errdef,npari,nparx,istat)
        ret = Struct(fmin=float(fmin),fedm=float(fedm),ferrdef=float(errdef),npari=int(npari),nparx=int(nparx),istat=int(istat))
        return ret
    
    def matrix_accurate(self):
        """check whether error matrix is accurate"""
        if self.tmin.fLimset: print "Warning: some parameter are up against limit"
        return self.mnstat().istat == 3
    
    def error_matrix(self):
        #void mnemat(Double_t* emat, Int_t ndim)
        pass
    def correlation_matrix(self):
        #void mnemat(Double_t* emat, Int_t ndim)
        pass
        
    def mnmatu(self):
        """print correlation coefficient"""
        return self.tmin.mnmatu(1)
    
    def help(self,cmd):
        """print out help"""
        self.tmin.mnhelp(cmd)

def main():
    def test(x,y):
        return (x-2)**2 + (y-3)**2 + 1.
    m = Minuit(test,x=2.1,y=2.9)
    m.set_up(1)
    m.set_strategy(2)
    
    m.migrad()
    m.minos()

    print m.values
    print m.errors
    m.mnmatu()
   
if __name__ == '__main__':
    main()