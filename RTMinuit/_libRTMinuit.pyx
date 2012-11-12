from libcpp.vector cimport vector
from libcpp.string cimport string
from cpython cimport exc
#from libcpp import bool
from util import *
from warnings import warn
from cython.operator cimport dereference as deref
from libc.math cimport sqrt
from pprint import pprint
include "Lcg_Minuit.pxi"

#our wrapper
cdef extern from "PythonFCN.h":
    #int raise_py_err()#this is very important we need custom error handler
    FunctionMinimum* call_mnapplication_wrapper(MnApplication app,unsigned int i, double tol) except +
    cdef cppclass PythonFCN(FCNBase):
        PythonFCN(object fcn, double up_parm, vector[string] pname,bint thrownan)
        double call "operator()" (vector[double] x) except +#raise_py_err
        double up()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

#look up map with default
cdef maplookup(m,k,d):
    return k[m] if k in m else d

cdef cfmin2struct(FunctionMinimum* cfmin):
    cfmin_struct = Struct(
            fval = cfmin.fval(),
            edm = cfmin.edm(),
            nfcn = cfmin.nfcn(),
            up = cfmin.up(),
            isValid = cfmin.isValid(),
            has_valid_parameters = cfmin.hasValidParameters(),
            has_accurate_covar = cfmin.hasAccurateCovar(),
            has_posdef_covar = cfmin.hasPosDefCovar(),
            has_made_posdef_covar = cfmin.hasMadePosDefCovar(),#forced to be posdef
            hesse_failed = cfmin.hesseFailed(),
            has_covariance = cfmin.hasCovariance(),
            is_above_max_edm = cfmin.isAboveMaxEdm(),
            has_reached_call_limit = cfmin.hasReachedCallLimit()
        )
    return cfmin_struct

cdef cfmin2covariance(FunctionMinimum* cfmin, int n):
    #not depending on numpy on purpose
    #cdef int n = cfmin.userState().params.size()
    return [[cfmin.userCovariance().get(i,j) for i in range(n)] for j in range(n)]


cdef cfmin2correlation(FunctionMinimum cfmin, int n):
    #cdef int n = cfmin.userState().params.size()
    #not depending on numpy on purpose
    return [[cfmin.userCovariance().get(i,j)/sqrt(cfmin.userCovariance().get(i,i))/sqrt(cfmin.userCovariance().get(j,j)) \
        for i in range(n)] for j in range(n)]


cdef class Minuit:
    cdef readonly object fcn
    cdef public object fitarg
    cdef readonly object narg
    cdef readonly object varname
    cdef readonly object pos2var
    cdef readonly object var2pos
    cdef public int strategy
    cdef public bint thrownan

    #PyMinuit Compatible interface
    cdef readonly object args
    cdef readonly object values
    cdef readonly object errors
    cdef readonly object covariance
    cdef readonly double fval
    cdef readonly double ncalls
    cdef readonly double edm
    cdef readonly object merrors

    #additional state variable
    cdef readonly object fmin
    cdef readonly int last_migrad_result
    cdef readonly int last
    cdef object initialvalue
    cdef object initialerror
    cdef object initiallimit
    cdef object initialfix

    cdef double errordef

    cdef PythonFCN* pyfcn
    cdef MnApplication* minimizer
    cdef FunctionMinimum* cfmin

    def __init__(self, fcn, thrownan=False, printmode=0, pedantic=True,
            errdef=1.0, strategy=1, **kwds):
        """
        construct minuit object
        arguments of f are pased automatically by the following order
        1) using f.func_code.co_varnames,f.func_code.co_argcount (all python function has this)
        2) using f.__call__.func_code.co_varnames, f.__call__.co_argcount (with self docked off)
        3) using inspect.getargspec(for some rare builtin function)

        user can set limit on paramater by passing limit_<varname>=(min,max) keyword argument
        user can set initial value onparameter by passing <varname>=value keyword argument
        user can fix parameter by doing fix_<varname>=True
        user can set initial step by passing error_<varname>=initialstep keyword argument

        if f_verbose is set to True FCN will be built for verbosity printing value and argument for every function call
        """

        args = better_arg_spec(fcn)
        narg = len(args)
        self.fcn = fcn

        #maintain 2 dictionary 1 is position to varname
        #and varname to position
        self.varname = args
        self.pos2var = {i: k for i, k in enumerate(args)}
        self.var2pos = {k: i for i, k in enumerate(args)}

        #self.set_printlevel(printlevel)
        #self.prepare(**kwds)

        self.last_migrad_result = 0
        self.args, self.values, self.errors = None, None, None

        if pedantic: self.pedantic(kwds)

        self.initialvalue = {x:maplookup(kwds,x,0.) for x in self.varname}
        self.initialerror = {x:maplookup(kwds,'error_'+x,1.) for x in self.varname}
        self.initiallimit = {x:maplookup(kwds,'limit_'+x,None) for x in self.varname}
        self.initialfix = {x:maplookup(kwds,'fix_'+x,False) for x in self.varname}

        self.pyfcn = NULL
        self.minimizer = NULL
        self.cfmin = NULL

        self.strategy = strategy
        self.errordef = errdef

    cdef construct_FCN(self):
        self.pyfcn = new PythonFCN(self.fcn,self.errordef,self.varname,False)

    def is_clean_state(self):
        return self.pyfcn is NULL and self.minimizer is NULL and self.cfmin is NULL

    cdef void clear_cobj(self):

        del self.pyfcn
        del self.minimizer
        del self.cfmin

        self.pyfcn = NULL
        self.minimizer = NULL
        self.cfmin = NULL


    def __dealloc__(self):
        self.clear_cobj()

    def pedantic(self, kwds):
        for vn in self.varname:
            if vn not in kwds:
                warn('Parameter %s does not have initial value. Assume 0.' % (vn))
            if 'error_'+vn not in kwds and 'fix_'+param_name(vn) not in kwds:
                warn('Parameter %s is floating but does not have initial step size. Assume 1.' % (vn))
        for vlim in extract_limit(kwds):
            if param_name(vlim) not in self.varname:
                warn('%s is given. But there is no parameter %s.Ignore.' % (vlim, param_name(vlim)))
        for vfix in extract_fix(kwds):
            if param_name(vfix) not in self.varname:
                warn('%s is given. But there is no parameter %s.Ignore.' % (vfix, param_name(vfix)))
        for verr in extract_error(kwds):
            if param_name(verr) not in self.varname :
                warn('%s float. But there is no parameter %s.Ignore.' % (verr, param_name(verr)))

    def refreshInternalState(self):
        #this is only to keep backward compatible with PyMinuit
        #it should be in a function instead of a state for lazy-callable
        pass

    cdef MnUserParameterState* initialParameterState(self):
        cdef MnUserParameterState* ret = new MnUserParameterState()
        cdef double lb
        cdef double ub
        for v in self.varname:
            ret.add(v,self.initialvalue[v],self.initialerror[v])
        for v in self.varname:
            if self.initiallimit[v] is not None:
                lb,ub = self.initiallimit[v]
                ret.setLimits(v,lb,ub)
        for v in self.varname:
            if self.initialfix[v]:
                ret.fix(v)
        return ret

    def migrad(self,int ncall=1000,resume=True,double tolerance=0.1, print_interval=100, print_at_the_end=True):
        """
            run migrad
            user can check if the return status is not 0
        """
        #construct new fcn and migrad if
        #it's a clean state or resume=False
        cdef MnUserParameterState* ups = NULL
        cdef MnStrategy* strat = NULL

        if not resume or self.is_clean_state():
            self.construct_FCN()
            if self.minimizer is not NULL: del self.minimizer
            ups = self.initialParameterState()
            strat = new MnStrategy(self.strategy)
            self.minimizer = new MnMigrad(deref(self.pyfcn),deref(ups),deref(strat))
            del ups; ups=NULL
            del strat; strat=NULL

        if self.cfmin is not NULL: print 'before ncall', self.cfmin.nfcn()
        del self.cfmin #remove the old one
        #this returns a real object need to copy
        self.cfmin = call_mnapplication_wrapper(deref(self.minimizer),ncall,tolerance)
        print 'after ncall', self.cfmin.nfcn()

        if print_at_the_end: self.print_cfmin(tolerance)

    def hesse(self):
        if self.cfmin is NULL:
            raise 'Run migrad or some minimizer first'

    def html_state(self):
        pass

    def print_cfmin(self,tolerance):
        cdef MnUserParameterState ust = MnUserParameterState(self.cfmin.userState())
        fmin = cfmin2struct(self.cfmin)
        self.print_cfmin_only(tolerance)
        self.print_state(ust)

    cdef print_state(self,MnUserParameterState upst):
        cdef vector[MinuitParameter] mps = upst.minuitParameters()
        cdef int i

        vnames=list()
        values=list()
        errs=list()
        lim_minus = list()
        lim_plus = list()
        fixstate = list()
        for i in range(mps.size()):
            vnames.append(mps[i].name())
            values.append(mps[i].value())
            errs.append(mps[i].error())
            fixstate.append(mps[i].isFixed())
            lim_plus.append(mps[i].upperLimit() if mps[i].hasUpperLimit() else None)
            lim_minus.append(mps[i].lowerLimit() if mps[i].hasLowerLimit() else None)

        self.print_state_template(vnames, values, errs, lim_minus = lim_minus, lim_plus = lim_plus, fixstate = fixstate)

    def print_initial_state(self):
        pass

    def print_cfmin_only(self,tolerance=None, ncalls = None):
        fmin = cfmin2struct(self.cfmin)
        goaledm = 0.001*tolerance*fmin.up if tolerance is not None else ''
        #http://wwwasdoc.web.cern.ch/wwwasdoc/hbook_html3/node125.html
        flatlocal = dict(locals().items()+fmin.__dict__.items())
        info1 = 'fval = %(fval)r | nfcn = %(nfcn)r | ncalls = %(ncalls)r\n'%flatlocal
        info2 = 'edm = %(edm)r (Goal: %(goaledm)r) | up = %(up)r\n'%flatlocal
        header1 = '|' + (' %14s |'*5)%('Valid','Valid Param','Accurate Covar','Posdef','Made Posdef')+'\n'
        hline = '-'*len(header1)+'\n'
        status1 = '|' + (' %14r |'*5)%(fmin.isValid, fmin.has_valid_parameters,
                fmin.has_accurate_covar,fmin.has_posdef_covar,fmin.has_made_posdef_covar)+'\n'
        header2 = '|' + (' %14s |'*5)%('Hesse Fail','Has Cov','Above EDM','','Reach calllim')+'\n'
        status2 = '|' + (' %14r |'*5)%(fmin.hesse_failed, fmin.has_covariance,
                fmin.is_above_max_edm,'',fmin.has_reached_call_limit)+'\n'

        print hline + info1 + info2 +\
            hline + header1 + hline + status1 +\
            hline + header2 + hline+ status2 +\
            hline


    def print_state_template(self,vnames, values, errs, minos_minus=None, minos_plus=None,
            lim_minus=None, lim_plus=None, fixstate=None):
        #for anyone using terminal
        maxlength = max([len(x) for x in vnames])
        maxlength = max(5,maxlength)

        header = ('| {:^4s} | {:^%ds} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} |\n'%maxlength).format(
                    '','Name', 'Value','Para Err', "Err-","Err+","Limit-","Limit+"," ")
        hline = '-'*len(header)+'\n'
        linefmt = '| {:>4d} | {:>%ds} = {:<8s} ± {:<8s} | {:<8s} | {:<8s} | {:<8s} | {:<8s} | {:^8s} |\n'%maxlength
        nfmt = '{:< 8.4G}'
        blank = ' '*8

        ret = hline+header+hline
        for i,v in enumerate(vnames):
            allnum = [i,v]
            for n in [values,errs,minos_minus,minos_plus,lim_minus,lim_plus]:
                if n is not None and n[i] is not None:
                    allnum+=[nfmt.format(n[i])]
                else:
                    allnum+=[blank]
            if fixstate is not None: 
                allnum += ['FIXED' if fixstate[i] else ' ']
            else:
                allnum += ['']
            line = linefmt.format(*allnum)
            ret+=line
        ret+=hline
        print ret
        #internally PyRoot store 1 FCN globally
        #so we need to change it to the correct one every time
        #It's limitation of C++
        # self.tmin.SetFCN(self.fcn)
        # self.last_migrad_result = self.tmin.Migrad()
        # self.set_ave()
        # return self.last_migrad_result

    def scan(self):
        raise NotImplementedError

    def contour(self):
        raise NotImplementedError

    def matrix(self):
        raise NotImplementedError


    # def prepare(self, **kwds):
    #     self.tmin.SetFCN(self.fcn)
    #     self.fix_param = []
    #     self.free_param = []
    #     for i, varname in self.pos2var.items():
    #         initialvalue = kwds[varname] if varname in kwds else 0.
    #         initialstep = kwds['error_' + varname] if 'error_' + varname in kwds else 0.1
    #         lrange, urange = kwds['limit_' + varname] if 'limit_' + varname in kwds else (0., 0.)
    #         ierflg = self.tmin.DefineParameter(i, varname, initialvalue, initialstep, lrange, urange)
    #         assert(ierflg == 0)
    #         #now fix parameter
    #     for varname in self.varname:
    #         if 'fix_' + varname in kwds and kwds['fix_'+varname]:
    #             self.tmin.FixParameter(self.var2pos[varname])
    #             self.fix_param.append(varname)
    #         else:
    #             self.free_param.append(varname)


    # def set_up(self, up):
    #     """set UP parameter 1 for chi^2 and 0.5 for log likelihood"""
    #     return self.errordef=up


    # def set_printlevel(self, lvl):
    #     """
    #     set printlevel -1 quiet, 0 normal, 1 verbose
    #     """
    #     return self.tmin.SetPrintLevel(lvl)


    # def set_strategy(self, strategy):
    #     """
    #     set strategy
    #     """
    #     return self.tmin.Command('SET STR %d' % strategy)


    # def command(self, cmd):
    #     """execute a command"""
    #     return self.tmin.Command(cmd)



    # def migrad_ok(self):
    #     """check whether last migrad call result is OK"""
    #     return self.last_migrad_result == 0


    # def hesse(self):
    #     """run hesse"""
    #     self.tmin.SetFCN(self.fcn)
    #     self.tmin.mnhess()
    #     self.set_ave()


    # def minos(self, varname=None):
    #     """run minos"""
    #     self.tmin.SetFCN(self.fcn)
    #     if varname is None:
    #         self.tmin.mnmnos()
    #     else:
    #         val2pl = ROOT.Double(0.)
    #         val2pi = ROOT.Double(0.)
    #         pos = self.var2pos[varname] + 1
    #         self.tmin.mnmnot(pos, 0, val2pl, val2pi)
    #     self.set_ave()


    # def set_ave(self):
    #     """set args values and errors"""
    #     tmp_values = {}
    #     tmp_errors = {}
    #     for i, varname in self.pos2var.items():
    #         tmp_val = ROOT.Double(0.)
    #         tmp_err = ROOT.Double(0.)
    #         self.tmin.GetParameter(i, tmp_val, tmp_err)
    #         tmp_values[varname] = float(tmp_val)
    #         tmp_errors[varname] = float(tmp_err)
    #     self.values = tmp_values
    #     self.errors = tmp_errors

    #     val = self.values
    #     tmparg = []
    #     for arg in self.varname:
    #         tmparg.append(val[arg])
    #     self.args = tuple(tmparg)
    #     self.fitarg.update(self.values)
    #     for k, v in self.errors.items():
    #         self.fitarg['error_' + k] = v


    # def mnstat(self):
    #     """
    #     return named tuple of cfmin,fedm,errdef,npari,nparx,istat
    #     """
    #     cfmin = ROOT.Double(0.)
    #     fedm = ROOT.Double(0.)
    #     errdef = ROOT.Double(0.)
    #     npari = ROOT.Long(0.)
    #     nparx = ROOT.Long(0.)
    #     istat = ROOT.Long(0.)
    #     #void mnstat(Double_t& cfmin, Double_t& fedm, Double_t& errdef, Int_t& npari, Int_t& nparx, Int_t& istat)
    #     self.tmin.mnstat(cfmin, fedm, errdef, npari, nparx, istat)
    #     ret = Struct(cfmin=float(cfmin), fedm=float(fedm), ferrdef=float(errdef), npari=int(npari), nparx=int(nparx),
    #         istat=int(istat))
    #     return ret


    # def cfmin(self):
    #     return self.mnstat().cfmin


    # def matrix_accurate(self):
    #     """check whether error matrix is accurate"""
    #     if self.tmin.fLimset: print "Warning: some parameter are up against limit"
    #     return self.mnstat().istat == 3


    # def error_matrix(self, correlation=False):
    #     ndim = self.mnstat().npari
    #     #void mnemat(Double_t* emat, Int_t ndim)
    #     tmp = array('d', [0.] * (ndim * ndim))
    #     self.tmin.mnemat(tmp, ndim)
    #     ret = np.array(tmp)
    #     ret = ret.reshape((ndim, ndim))
    #     if correlation:
    #         diag = np.diagonal(ret)
    #         sigma_col = np.sqrt(diag[:, np.newaxis])
    #         sigma_row = sigma_col.T
    #         ret = ret / sigma_col / sigma_row
    #     return ret


    # def mnmatu(self):
    #     """print correlation coefficient"""
    #     return self.tmin.mnmatu(1)


    # def help(self, cmd):
    #     """print out help"""
    #     self.tmin.mnhelp(cmd)


    # def minos_errors(self):
    #     ret = {}
    #     self.tmin.SetFCN(self.fcn)
    #     for i, v in self.pos2var.items():
    #         eplus = ROOT.Double(0.)
    #         eminus = ROOT.Double(0.)
    #         eparab = ROOT.Double(0.)
    #         gcc = ROOT.Double(0.)
    #         self.tmin.mnerrs(i, eplus, eminus, eparab, gcc)
    #         #void mnerrs(Int_t number, Double_t& eplus, Double_t& eminus, Double_t& eparab, Double_t& gcc)
    #         ret[v] = Struct(eplus=float(eplus), eminus=float(eminus), eparab=float(eparab), gcc=float(gcc))
    #     return ret

    # def html_results(self):
    #     return MinuitHTMLResult(self)

    # def list_of_fixed_param(self):
    #     tmp_ret = list()#fortran index
    #     for i in range(self.tmin.GetNumFixedPars()):
    #         tmp_ret.append(self.tmin.fIpfix[i])
    #     #now get the constants
    #     for i in range(self.tmin.GetNumPars()):
    #         if self.tmin.fNvarl[i] == 0:
    #             tmp_ret.append(i+1)
    #     tmp_ret = list(set(tmp_ret))
    #     tmp_ret.sort()
    #     for i in range(len(tmp_ret)):
    #         tmp_ret[i]-=1 #convert to position
    #     ret = [self.pos2var[x] for x in tmp_ret]
    #     return ret

    # def list_of_vary_param(self):
    #     fix_pars = self.list_of_fixed_param()
    #     ret = [v for v in self.varname if v not in fix_pars]
    #     return ret

    # def html_error_matrix(self):
    #     return MinuitCorrelationMatrixHTML(self)

