__all__ = ['Minuit']
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
    FunctionMinimum* call_mnapplication_wrapper(\
        MnApplication app,unsigned int i, double tol) except +
    cdef cppclass PythonFCN(FCNBase):
        PythonFCN(\
            object fcn, double up_parm, vector[string] pname,bint thrownan)
        double call "operator()" (vector[double] x) except +#raise_py_err
        double up()
        int getNumCall()
        void resetNumCall()

#look up map with default
cdef maplookup(m,k,d):
    return m[k] if k in m else d


cdef cfmin2struct(FunctionMinimum* cfmin):
    cfmin_struct = Struct(
            fval = cfmin.fval(),
            edm = cfmin.edm(),
            nfcn = cfmin.nfcn(),
            up = cfmin.up(),
            is_valid = cfmin.isValid(),
            has_valid_parameters = cfmin.hasValidParameters(),
            has_accurate_covar = cfmin.hasAccurateCovar(),
            has_posdef_covar = cfmin.hasPosDefCovar(),
            #forced to be posdef
            has_made_posdef_covar = cfmin.hasMadePosDefCovar(),
            hesse_failed = cfmin.hesseFailed(),
            has_covariance = cfmin.hasCovariance(),
            is_above_max_edm = cfmin.isAboveMaxEdm(),
            has_reached_call_limit = cfmin.hasReachedCallLimit()
        )
    return cfmin_struct

cdef minuitparam2struct(MinuitParameter* mp):
    ret = Struct(
            number = mp.number(),
            naem = mp.name(),
            value = mp.value(),
            error = mp.error(),
            is_const = mp.isConst(),
            is_fixed = mp.isFixed(),
            has_limits = mp.hasLimits(),
            has_lower_limit = mp.hasLowerLimit(),
            has_upper_limit = mp.hasUpperLimit(),
            lower_limit = mp.lowerLimit(),
            upper_limit = mp.upperLimit(),
        )
    return ret

cdef cfmin2covariance(FunctionMinimum cfmin, int n):
    #not depending on numpy on purpose
    #cdef int n = cfmin.userState().params.size()
    return [[cfmin.userCovariance().get(i,j) \
        for i in range(n)] for j in range(n)]


cdef cfmin2correlation(FunctionMinimum cfmin, int n):
    #cdef int n = cfmin.userState().params.size()
    #not depending on numpy on purpose
    return [[cfmin.userCovariance().get(i,j)/\
        sqrt(cfmin.userCovariance().get(i,i))/\
        sqrt(cfmin.userCovariance().get(j,j)) \
        for i in range(n)] for j in range(n)]

cdef minoserror2struct(MinosError m):
        ret = Struct(
            lower = m.lower(),
            upper = m.upper(),
            is_valid = m.isValid(),
            lower_valid = m.lowerValid(),
            upper_valid = m.upperValid(),
            at_lower_limit = m.atLowerLimit(),
            at_upper_limit = m.atUpperLimit(),
            at_lower_max_fcn = m.atLowerMaxFcn(),
            at_upper_max_fcn = m.atUpperMaxFcn(),
            lower_new_min = m.lowerNewMin(),
            upper_new_min = m.upperNewMin(),
            nfcn = m.nfcn(),
            min = m.min()
            )
        return ret


cdef class Minuit:
    #standard stuff
    cdef readonly object fcn #:fcn
    #cdef readonly object varname #:variable names
    cdef readonly object pos2var#:map variable position to varname
    cdef readonly object var2pos#:map varname to position

    #Initial settings
    cdef object initialvalue #:hold initial values
    cdef object initialerror #:hold initial errors
    cdef object initiallimit #:hold initial limits
    cdef object initialfix #:hold initial fix state

    #C++ object state
    cdef PythonFCN* pyfcn #:FCN
    cdef MnApplication* minimizer #:migrad
    cdef FunctionMinimum* cfmin #:last migrad result
    #:last parameter state(from hesse/migrad)
    cdef MnUserParameterState* last_upst

    #PyMinuit compatible field
    cdef public double up #:UP parameter
    cdef public double tol #:tolerance migrad stops when edm>0.0001*tol*UP
    cdef public unsigned int strategy #:0 fast 1 default 2 slow but accurate
    #: 0: quiet 1: print stuff the end 2: 1+fit status during call
    #: yes I know the case is wrong but this is to keep it compatible with
    #: PyMinuit
    cdef public printMode
    #: raise runtime error if function evaluate to nan
    cdef readonly bint throw_nan

    #PyMinuit Compatible interface
    cdef readonly object parameters#:tuple of parameter name(correct order)
    cdef readonly object args#:tuple of values
    cdef readonly object values#:map varname -> value
    cdef readonly object errors#:map varname -> parabolic error
    cdef readonly object covariance#:map (v1,v2)->covariance
    cdef readonly double fval#:last value of fcn
    cdef readonly double ncalls#:number of fcn call of last migrad/minos/hesse
    cdef readonly double edm#Estimate distance to minimum
    #:minos error ni a funny map from
    #:(vname,direction)->error
    #:direction is 1.0 for positive error and -1.0 for negative error
    cdef readonly object merrors
    #:global correlation coefficient
    cdef readonly object gcc
    #and some extra
    #:map of
    #:varname -> value
    #:error_varname -> error
    #:limit_varname -> limit
    #:fix_varname -> True/False
    #:user can just use python keyword expansion to use all the argument like
    #:Minuit(fcn,**fitargs)
    cdef public object fitarg
    cdef readonly object narg#: number of arguments
    #: map vname-> struct with various minos error calcumation information
    cdef public object merrors_struct
    cdef public object printer

    def __init__(self, fcn,
            throw_nan=False,  pedantic=True,
            printMode=0, printer=None, **kwds):
        """
        construct minuit object
        arguments of f are pased automatically by the following order
        1) using f.func_code.co_varnames,f.func_code.co_argcount
        (all python function has this)
        2) using f.__call__.func_code.co_varnames, f.__call__.co_argcount
        (with self docked off)
        3) using inspect.getargspec(for some rare builtin function)

        user can set limit on paramater by passing
        limit_<varname>=(min,max) keyword argument
        user can set initial value onparameter by passing
        <varname>=value keyword argument
        user can fix parameter by doing
        fix_<varname>=True
        user can set initial step by passing
        error_<varname>=initialstep keyword argument

        if f_verbose is set to True FCN will be built for verbosity
        printing value and argument for every function call
        """

        args = better_arg_spec(fcn)
        narg = len(args)
        self.fcn = fcn

        #maintain 2 dictionary 1 is position to varname
        #and varname to position
        #self.varname = args
        self.pos2var = {i: k for i, k in enumerate(args)}
        self.var2pos = {k: i for i, k in enumerate(args)}

        self.args, self.values, self.errors = None, None, None

        self.initialvalue = {x:maplookup(kwds,x,0.) for x in args}
        self.initialerror = \
            {x:maplookup(kwds,'error_'+x,1.) for x in args}
        self.initiallimit = \
            {x:maplookup(kwds,'limit_'+x,None) for x in args}
        self.initialfix = \
            {x:maplookup(kwds,'fix_'+x,False) for x in args}

        self.pyfcn = NULL
        self.minimizer = NULL
        self.cfmin = NULL
        self.last_upst = NULL

        self.up = 1.0
        self.tol = 0.1
        self.strategy = 1
        self.printMode = printMode
        self.throw_nan = throw_nan

        self.parameters = args
        self.args = None
        self.values = None
        self.errors = None
        self.covariance = None
        self.fval = 0.
        self.ncalls = 0
        self.edm = 1.
        self.merrors = {}
        self.gcc = {}
        if pedantic: self.pedantic(kwds)

        self.fitarg = {}
        self.fitarg.update(self.initialvalue)
        self.fitarg.update(
            {'error_'+k:v for k,v in self.initialerror.items()})
        self.fitarg.update(
            {'limit_'+k:v for k,v in self.initiallimit.items()})
        self.fitarg.update(
            {'fix_'+k:v for k,v in self.initialfix.items()})

        self.narg = len(self.parameters)

        self.merrors_struct = {}


    cdef construct_FCN(self):
        """(re)construct FCN"""
        del self.pyfcn
        self.pyfcn = new PythonFCN(
                self.fcn,
                self.up,
                self.parameters,
                self.throw_nan)


    def is_clean_state(self):
        """check if minuit is at clean state ie. no migrad call"""
        return self.pyfcn is NULL and \
            self.minimizer is NULL and self.cfmin is NULL


    cdef void clear_cobj(self):
        #clear C++ internal state
        del self.pyfcn;self.pyfcn = NULL
        del self.minimizer;self.minimizer = NULL
        del self.cfmin;self.cfmin = NULL
        del self.last_upst;self.last_upst = NULL


    def __dealloc__(self):
        self.clear_cobj()


    def pedantic(self, kwds):
        for vn in self.parameters:
            if vn not in kwds:
                warn(('Parameter %s does not have initial value. '
                    'Assume 0.') % (vn))
            if 'error_'+vn not in kwds and 'fix_'+param_name(vn) not in kwds:
                warn(('Parameter %s is floating but does not '
                    'have initial step size. Assume 1.') % (vn))
        for vlim in extract_limit(kwds):
            if param_name(vlim) not in self.parameters:
                warn(('%s is given. But there is no parameter %s. '
                    'Ignore.') % (vlim, param_name(vlim)))
        for vfix in extract_fix(kwds):
            if param_name(vfix) not in self.parameters:
                warn(('%s is given. But there is no parameter %s. \
                    Ignore.') % (vfix, param_name(vfix)))
        for verr in extract_error(kwds):
            if param_name(verr) not in self.parameters:
                warn(('%s float. But there is no parameter %s. \
                    Ignore.') % (verr, param_name(verr)))


    cdef refreshInternalState(self):
        """
        refresh internal state attributes.
        These attributes should be in a function instead
        but kept here for PyMinuit compatiblity
        """
        cdef vector[MinuitParameter] mpv
        cdef MnUserCovariance cov
        if self.last_upst is not NULL:
            mpv = self.last_upst.minuitParameters()
            self.values = {}
            self.errors = {}
            self.args = []
            for i in range(mpv.size()):
                self.args.append(mpv[i].value())
                self.values[mpv[i].name()] = mpv[i].value()
                self.errors[mpv[i].name()] = mpv[i].value()
            self.args = tuple(self.args)
            self.fitarg.update(self.values)
            cov = self.last_upst.covariance()
            self.covariance =\
                {(self.parameters[i],self.parameters[j]):cov.get(i,j)\
                    for i in range(self.narg) for j in range(self.narg)}
            self.fval = self.last_upst.fval()
            self.ncalls = self.last_upst.nfcn()
            self.edm = self.last_upst.edm()
            self.gcc = {v:self.last_upst.globalCC().globalCC()[i]\
                        for i,v in enumerate(self.parameters)}
        self.merrors = {(k,1.0):v.upper
                       for k,v in self.merrors_struct.items()}
        self.merrors.update({(k,-1.0):v.lower
                       for k,v in self.merrors_struct.items()})


    cdef MnUserParameterState* initialParameterState(self):
        """
        construct parameter state from initial array.
        caller is responsible for cleaning up the pointer 
        """
        cdef MnUserParameterState* ret = new MnUserParameterState()
        cdef double lb
        cdef double ub
        for v in self.parameters:
            ret.add(v,self.initialvalue[v],self.initialerror[v])
        for v in self.parameters:
            if self.initiallimit[v] is not None:
                lb,ub = self.initiallimit[v]
                ret.setLimits(v,lb,ub)
        for v in self.parameters:
            if self.initialfix[v]:
                ret.fix(v)
        return ret


    def migrad(self,int ncall=1000,resume=True,double tolerance=0.1,
            print_interval=100, print_at_the_end=True):
        """
        run migrad, the age-tested(stable over 20 years old),
        super robust minizer.
        """
        #construct new fcn and migrad if
        #it's a clean state or resume=False
        cdef MnUserParameterState* ups = NULL
        cdef MnStrategy* strat = NULL
        self.print_banner('MIGRAD')
        if not resume or self.is_clean_state():
            self.construct_FCN()
            if self.minimizer is not NULL: del self.minimizer
            ups = self.initialParameterState()
            strat = new MnStrategy(self.strategy)
            self.minimizer = \
                    new MnMigrad(deref(self.pyfcn),deref(ups),deref(strat))
            del ups; ups=NULL
            del strat; strat=NULL

        del self.cfmin #remove the old one
        #this returns a real object need to copy
        self.cfmin = call_mnapplication_wrapper(
                deref(self.minimizer),ncall,tolerance)
        del self.last_upst
        self.last_upst = new MnUserParameterState(self.cfmin.userState())
        self.refreshInternalState()
        if print_at_the_end: self.print_cfmin(tolerance)


    def hesse(self):
        """
        run HESSE.
        HESSE estimate error by the second derivative at the minimim.
        """

        cdef MnHesse* hesse = NULL
        cdef MnUserParameterState upst
        self.print_banner('HESSE')
        #if self.cfmin is NULL:
            #raise RuntimeError('Run migrad first')
        hesse = new MnHesse(self.strategy)
        upst = hesse.call(deref(self.pyfcn),self.cfmin.userState())

        del self.last_upst
        self.last_upst = new MnUserParameterState(upst)
        self.refreshInternalState()
        del hesse


    def minos(self, var = None, sigma = 1,
                unsigned int maxcall=1000):
        """
        run minos for paramter *var* n *sigma* uncertainty.
        If *var* is None it runs minos for all parameters
        """
        cdef unsigned int index = 0
        cdef MnMinos* minos = NULL
        cdef MinosError mnerror
        cdef char* name = NULL
        self.print_banner('MINOS')
        if var is not None:
            name = var
            index = self.cfmin.userState().index(var)
            if self.cfmin.userState().minuitParameters()[i].isFixed():
                return None
            minos = new MnMinos(deref(self.pyfcn), deref(self.cfmin),strategy)
            mnerror = minos.minos(index,maxcall)
            self.merrors_struct[var]=minoserror2struct(mnerror)
            self.print_mnerror(var,self.merrors_struct[var])
        else:
            for vname in self.parameters:
                index = self.cfmin.userState().index(vname)
                if self.cfmin.userState().minuitParameters()[index].isFixed():
                    continue
                minos = \
                    new MnMinos(deref(self.pyfcn), deref(self.cfmin),strategy)
                mnerror = minos.minos(index,maxcall)
                self.merrors_struct[vname]=minoserror2struct(mnerror)
                self.print_mnerror(vname,self.merrors_struct[vname])
        self.refreshInternalState()
        del minos
        return self.merrors_struct


    def matrix(self, correlation=False, skip_fixed=False):
        """return error/correlation matrix in tuple or tuple format."""
        if self.last_upst is NULL:
            raise RuntimeError("Run migrad/hesse first")
        cdef MnUserCovariance cov = self.last_upst.covariance()
        ret = tuple(
                tuple(cov.get(iv1,iv2)
                    for iv1,v1 in self.parameters \
                        if not skip_fixed or self.is_fixed(v1))
                    for iv2,v2 in enumerate(self.parameters) \
                        if not skip_fixed or self.is_fixed(v2)
                )
        return ret


    def np_matrix(self, correlation=False, skip_fixed=False):
        """return error/correlation matrix in numpy array format."""
        import numpy as np
        #TODO make a not so lazy one
        return np.array(matrix)

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

    def is_fixed(self,vname):
        if vname not in self.parameters:
            raise RuntimeError('Cannot find %s in list of variables.')
        cdef unsigned int index = var2pos[vname]
        if self.last_upst is NULL:
            return self.initialfix[vname]
        else:
            return self.last_upst.minuitParameters()[index].isFixed()


    def scan(self):
        #anyone actually use this?
        raise NotImplementedError


    def contour(self):
        #and this?
        raise NotImplementedError

    #TODO: Modularize this
    #######Terminal Display Stuff######################
    #This is 2012 USE IPYTHON PEOPLE!!! :P
    def print_cfmin(self,tolerance):
        cdef MnUserParameterState ust = MnUserParameterState(
                                        self.cfmin.userState())
        ncalls = 0 if self.pyfcn is NULL else self.pyfcn.getNumCall()
        fmin = cfmin2struct(self.cfmin)
        print '*'*30
        self.print_cfmin_only(tolerance,ncalls)
        self.print_state(ust)
        print '*'*30


    def print_mnerror(self,vname,smnerr):
        stat = 'VALID' if smnerr.is_valid else 'PROBLEM'

        summary = 'Minos Status for %s: %s\n'%\
                (vname,stat)

        error = '| {:^15s} | {: >12g} | {: >12g} |\n'.format(
                    'Error',
                    smnerr.
                    lower,
                    smnerr.upper)
        valid = '| {:^15s} | {:^12s} | {:^12s} |\n'.format(
                    'Valid',
                    str(smnerr.lower_valid),
                    str(smnerr.upper_valid))
        at_limit='| {:^15s} | {:^12s} | {:^12s} |\n'.format(
                    'At Limit',
                    str(smnerr.at_lower_limit),
                    str(smnerr.at_upper_limit))
        max_fcn='| {:^15s} | {:^12s} | {:^12s} |\n'.format(
                    'Max FCN',
                    str(smnerr.at_lower_max_fcn),
                    str(smnerr.at_upper_max_fcn))
        new_min='| {:^15s} | {:^12s} | {:^12s} |\n'.format(
                    'New Min',
                    str(smnerr.lower_new_min),
                    str(smnerr.upper_new_min))
        hline = '-'*len(error)+'\n'
        print hline +\
              summary +\
              hline +\
              error +\
              valid +\
              at_limit +\
              max_fcn +\
              new_min +\
              hline


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
            lim_plus.append(
                mps[i].upperLimit() if mps[i].hasUpperLimit() else None)
            lim_minus.append(
                mps[i].lowerLimit() if mps[i].hasLowerLimit() else None)

        self.print_state_template(
                vnames,
                values,
                errs,
                lim_minus = lim_minus,
                lim_plus = lim_plus,
                fixstate = fixstate)


    def print_initial_state(self):
        raise NotImplementedError


    def print_cfmin_only(self,tolerance=None, ncalls = 0):
        fmin = cfmin2struct(self.cfmin)
        goaledm = 0.0001*tolerance*fmin.up if tolerance is not None else ''
        #despite what the doc said the code is actually 1e-4
        #http://wwwasdoc.web.cern.ch/wwwasdoc/hbook_html3/node125.html
        flatlocal = dict(locals().items()+fmin.__dict__.items())
        info1 = 'fval = %(fval)r | nfcn = %(nfcn)r | ncalls = %(ncalls)r\n'%\
                flatlocal
        info2 = 'edm = %(edm)r (Goal: %(goaledm)r) | up = %(up)r\n'%flatlocal
        header1 = '|' + (' %14s |'*5)%(
                    'Valid',
                    'Valid Param',
                    'Accurate Covar',
                    'Posdef',
                    'Made Posdef')+'\n'
        hline = '-'*len(header1)+'\n'
        status1 = '|' + (' %14r |'*5)%(
                    fmin.is_valid,
                    fmin.has_valid_parameters,
                    fmin.has_accurate_covar,
                    fmin.has_posdef_covar,
                    fmin.has_made_posdef_covar)+'\n'
        header2 = '|' + (' %14s |'*5)%(
                    'Hesse Fail',
                    'Has Cov',
                    'Above EDM',
                    '',
                    'Reach calllim')+'\n'
        status2 = '|' + (' %14r |'*5)%(
                    fmin.hesse_failed,
                    fmin.has_covariance,
                    fmin.is_above_max_edm,
                    '',
                    fmin.has_reached_call_limit)+'\n'

        print hline + info1 + info2 +\
            hline + header1 + hline + status1 +\
            hline + header2 + hline+ status2 +\
            hline


    def print_state_template(self,vnames, values, errs, 
            minos_minus=None, minos_plus=None,
            lim_minus=None, lim_plus=None, fixstate=None):
        #for anyone using terminal
        maxlength = max([len(x) for x in vnames])
        maxlength = max(5,maxlength)

        header = ('| {:^4s} | {:^%ds} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} | {:^8s} |\n'%maxlength)\
                .format(
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


    def print_banner(self, cmd):
        ret = '*'*50+'\n'
        ret += '*{:^48}*'.format(cmd)+'\n'
        ret += '*'*50+'\n'
        print ret


    def print_all_minos(self,cmd):
        for vname in varnames:
            if vname in self.merrors_struct:
                self.print_mnerror(vname,self.merrors_struct[vname])


    def set_up(self, up):
        """set UP parameter 1 for chi^2 and 0.5 for log likelihood"""
        self.up = up


    def set_strategy(self,stra):
        """set strategy 0=fast , 1=default, 2=slow but accurate"""
        self.strategy=stra


    def set_print_mode(self, lvl):
        """set printlevel 0 quiet, 1 normal, 2 paranoid, 3 really paranoid """
        self.printMode = lvl


    def migrad_ok(self):
        """check if minimum is valid"""
        return self.cfmin is not NULL and self.fmin.isValid()


    def matrix_acurate(self):
        """check if covariance is accurate"""
        return self.last_upst is not NULL and self.last_upst.hasCovariance()


    def html_results(self):
        """show result in html form"""
        return MinuitHTMLResult(self)


    def html_error_matrix(self):
        """show error matrix in html form"""
        return MinuitCorrelationMatrixHTML(self)


    def list_of_fixed_param(self):
        """return list of (initially) fixed parameters"""
        return [v for v in self.parameters if self.initialfix[v]]


    def list_of_vary_param(self):
        """return list of (initially) float vary parameters"""
        return
