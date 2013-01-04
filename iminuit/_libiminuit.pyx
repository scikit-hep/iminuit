#cython: embedsignature=True
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
from ConsoleFrontend import ConsoleFrontend
from iminuit_warnings import *
import _plotting
include "Lcg_Minuit.pxi"
include "Minuit2Struct.pxi"
import array
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
        void set_up(double up)
        void resetNumCall()


#look up map with default
cdef maplookup(m,k,d):
    return m[k] if k in m else d


cdef class Minuit:
    #standard stuff

    cdef readonly object fcn #:fcn
    #cdef readonly object varname #:variable names
    """this should work"""
    cdef readonly object pos2var#:map variable position to varname
    """or this should work"""
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
    cdef public double errordef #:UP parameter
    cdef public double tol #:tolerance migrad stops when edm<0.0001*tol*UP
    cdef public unsigned int strategy #:0 fast 1 default 2 slow but accurate
    #: 0: quiet 1: print stuff the end 2: 1+fit status during call
    #: yes I know the case is wrong but this is to keep it compatible with
    #: PyMinuit
    cdef public print_level
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
    cdef public object frontend

    def __init__(self, fcn,
            throw_nan=False, pedantic=True,
            frontend=None, forced_parameters=None, print_level=1,
            errordef=None, **kwds):
        """
        Construct minuit object from given *fcn*

        **Arguments:**

            - **fcn**: function to optimized. Minuit automagically find how to
              call the function and each parameters. More information about how
              Minuit detects function signature can be found in
              :ref:`function-sig-label`

        **Builtin Keyword Arguments:**

            - **throw_nan**: fcn can be set to raise RuntimeError when it
              encounters *nan*. (Default False)

            - **pedantic**: warns about parameters that do not have initial
              value or initial error/stepsize set.

            - **frontend**: Minuit frontend. There are two builtin frontends.

                1. ConsoleFrontend which is design to print out to terminal.

                2. HtmlFrontend which is designed to give a nice output in
                   IPython notebook session.

              By Default, Minuit switch to HtmlFrontend automatically if it
              is called in IPython session. It uses ConsoleFrontend otherwise.

            - **forced_parameters**: tell Minuit not to do function signature
              detection and use this argument instead. (Default None
              (automagically detect signature)

            - **print_level**: set the print_level for this Minuit. 0 is quiet.
              1 print out at the end of migrad/hesse/minos. The reason it
              has this cAmEl case is to keep it compatible with PyMinuit.

            - **errordef**: Optionals. Amount of increase in fcn to be defined
              as 1 :math:`\sigma`. If None is given, it will look at
              `fcn.default_errordef()`. If `fcn.default_errordef()` is not defined or
              not callable iminuit will give a warning and set errordef to 1.
              Default None(which means errordef=1 with a warning).

        **Parameter Keyword Arguments:**

            Similar to PyMinuit. iminuit allows user to set initial value,
            initial stepsize/error, limits of parameters and whether
            parameter should be fixed or not by passing keyword arguments to
            Minuit. This is best explained through an example::

                def f(x,y):
                    return (x-2)**2 + (y-3)**2

            * Initial value(varname)::

                #initial value for x and y
                m = Minuit(f, x=1, y=2)

            * Initial step size/error(fix_varname)::

                #initial step size for x and y
                m = Minuit(f, error_x=0.5, error_y=0.5)

            * Limits (limit_varname=tuple)::

                #limits x and y
                m = Minuit(f, limit_x=(-10,10), limit_y=(-20,20))

            * Fixing parameters::

                #fix x but vary y
                m = Minuit(f, fix_x=True)

            .. note::

                Tips: you can use python dictionary expansion to
                programatically change fitting argument.

                ::

                    kwdarg = dict(x=1., error_x=0.5)
                    m = Minuit(f, **kwdarg)

                You can obtain also obtain fit arguments from Minuit object
                to reuse it later too. Note that fitarg will be automatically
                updated to minimum value and corresponding error when you ran
                migrad/hesse::

                    m = Minuit(f, x=1, error_x=0.5)
                    my_fitarg = m.fitarg
                    another_fit = Minuit(f, **my_fitarg)

        """

        args = describe(fcn) if forced_parameters is None\
               else forced_parameters
        self._check_extra_args(args,kwds)
        narg = len(args)
        self.fcn = fcn

        self.frontend = self._auto_frontend() if frontend is None else frontend

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

        if errordef is None:
            default_errordef = getattr(fcn,'default_errordef', None)
            if not callable(default_errordef):
                if pedantic:
                    warn(InitialParamWarning('errordef is not given. Default to 1.'))
                self.errordef = 1.0
            else:
                self.errordef = default_errordef()
        else:
            self.errordef = errordef
        self.tol = 0.1
        self.strategy = 1
        self.print_level = print_level
        set_migrad_print_level(print_level)
        self.throw_nan = throw_nan

        self.parameters = args
        self.args = tuple(self.initialvalue[k] for k in args)
        self.values = {k:self.initialvalue[k] for k in args}
        self.errors = {k:self.initialerror[k] for k in args}
        self.covariance = None
        self.fval = 0.
        self.ncalls = 0
        self.edm = 1.
        self.merrors = {}
        self.gcc = None
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


    def migrad(self, int ncall=10000, resume=True, int nsplit=1):
        """Run migrad.

        Migrad is an age-tested(over 40 years old, no kidding), super
        robust and stable minimization algorithm. It even has
        `wiki page <http://en.wikipedia.org/wiki/MINUIT>`_.
        You can read how it does the magic at
        `here <http://wwwasdoc.web.cern.ch/wwwasdoc/minuit/minmain.html>`_.

        **Arguments:**

            * **ncall**: integer (approximate) maximum number of call before
              migrad stop trying. Default 10000

            * **resume**: boolean indicating whether migrad should resume from
              previous minimizer attempt(True) or should start from the
              beginning(False). Default True.

            * **split**: split migrad in to *split* runs. Max fcn call
              for each run is ncall/nsplit. Migrad stops when it found the
              function minimum to be valid or ncall is reached. This is useful
              for getting progress. However, you need to make sure that
              ncall/nsplit is large enough. Otherwise, migrad will think
              that the minimum is invalid due to exceeding max call
              (ncall/nsplit). Default 1(no split).

        **Return:**

            :ref:`function-minimum-sruct`, list of :ref:`minuit-param-struct`
        """
        #construct new fcn and migrad if
        #it's a clean state or resume=False
        cdef MnUserParameterState* ups = NULL
        cdef MnStrategy* strat = NULL

        if self.print_level>0: self.frontend.print_banner('MIGRAD')

        if not resume or self.is_clean_state():
            self.construct_FCN()
            if self.minimizer is not NULL: del self.minimizer
            ups = self.initialParameterState()
            strat = new MnStrategy(self.strategy)
            self.minimizer = \
                    new MnMigrad(deref(self.pyfcn),deref(ups),deref(strat))
            del ups; ups=NULL
            del strat; strat=NULL

        if not resume: self.pyfcn.resetNumCall()

        del self.cfmin #remove the old one

        #this returns a real object need to copy
        ncall_round = round(1.0*(ncall)/nsplit)
        assert(ncall_round>0)
        totalcalls = 0
        first = True
        while (first) or \
                (not self.cfmin.isValid() and totalcalls < ncall):
            first=False
            self.cfmin = call_mnapplication_wrapper(
                    deref(self.minimizer),ncall_round,self.tol)
            del self.last_upst
            self.last_upst = new MnUserParameterState(self.cfmin.userState())
            totalcalls+=ncall_round#self.cfmin.nfcn()
            if self.print_level>1: self.print_fmin()

        del self.last_upst
        self.last_upst = new MnUserParameterState(self.cfmin.userState())
        self.refreshInternalState()

        if self.print_level>0: self.print_fmin()

        return self.get_fmin(), self.get_param_states()


    def hesse(self):
        """Run HESSE.

        HESSE estimates error matrix by the `second derivative at the minimim
        <http://en.wikipedia.org/wiki/Hessian_matrix>`_. This error matrix
        is good if your :math:`\chi^2` or likelihood profile is parabolic at
        the minimum. From my experience, most of simple fits are.

        :meth:`minos` makes no parabolic assumption and scan the likelihood
        and give the correct error asymmetric error in all cases(Unless your
        likelihood profile is utterly discontinuous near the minimum). But
        it is much more computationally expensive.

        **Returns**
            list of :ref:`minuit-param-struct`
        """

        cdef MnHesse* hesse = NULL
        cdef MnUserParameterState upst
        if self.print_level>0: self.frontend.print_banner('HESSE')
        if self.cfmin is NULL:
            raise RuntimeError('Run migrad first')
        hesse = new MnHesse(self.strategy)
        upst = hesse.call(deref(self.pyfcn),self.cfmin.userState())
        if not upst.hasCovariance():
            warn("HESSE Failed. Covariance and GlobalCC will not be available",
                HesseFailedWarning)
        del self.last_upst
        self.last_upst = new MnUserParameterState(upst)
        self.refreshInternalState()
        del hesse

        if self.print_level>0:
            self.print_param()
            self.print_matrix()

        return self.get_param_states()


    def minos(self, var = None, sigma = 1., unsigned int maxcall=1000):
        """Run minos for parameter *var*

        If *var* is None it runs minos for all parameters

        **Arguments:**

            - **var**: optional variable name. Default None.(run minos for
              every variable)
            - **sigma**: number of :math:`\sigma` error. Default 1.0.

        **Returns**

            Dictionary of varname to :ref:`minos-error-struct`
            if minos is requested for all parameters.

        """
        if self.pyfcn is NULL or self.cfmin is NULL:
            raise RuntimeError('Minos require function to be at the minimum. Run migrad first.')
        cdef unsigned int index = 0
        cdef MnMinos* minos = NULL
        cdef MinosError mnerror
        cdef char* name = NULL
        cdef double oldup = self.pyfcn.up()
        self.pyfcn.set_up(oldup*sigma*sigma)
        if self.print_level>0: self.frontend.print_banner('MINOS')
        if not self.cfmin.isValid():
            raise RuntimeError(('Function mimimum is not valid. Make sure'
                ' migrad converge first'))
        if var is not None and var not in self.parameters:
                raise RuntimeError('Specified parameters(%r) cannot be found'
                    ' in parameter list :'%var+str(self.parameters))

        varlist = [var] if var is not None else self.parameters

        fixed_param = self.list_of_fixed_param()
        for vname in varlist:
            index = self.cfmin.userState().index(vname)

            if vname in fixed_param:
                if var is not None:#specifying vname but it's fixed
                    warnings.warn(RuntimeWarning('Specified variable name for minos is set to fixed'))
                    return None
                continue
            minos = new MnMinos(deref(
                self.pyfcn), deref(self.cfmin),self.strategy)
            mnerror = minos.minos(index,maxcall)
            self.merrors_struct[vname]=minoserror2struct(mnerror)
            if self.print_level>0:
                self.frontend.print_merror(
                    vname,self.merrors_struct[vname])
        self.refreshInternalState()
        del minos
        self.pyfcn.set_up(oldup)
        return self.merrors_struct


    def matrix(self, correlation=False, skip_fixed=True):
        """return error/correlation matrix in tuple or tuple format."""
        if self.last_upst is NULL:
            raise RuntimeError("Run migrad/hesse first")
        if not skip_fixed:
            raise RunTimeError('skip_fixed=False is not supported')
        if not self.last_upst.hasCovariance():
            raise RuntimeError("Covariance is not valid. May be the last Hesse call failed?")

        cdef MnUserCovariance cov = self.last_upst.covariance()
        params = self.list_of_vary_param()
        if correlation:
            ret = tuple(
                tuple(cov.get(iv1,iv2)/sqrt(cov.get(iv1,iv1)*cov.get(iv2,iv2))
                    for iv1,v1 in enumerate(params))\
                    for iv2,v2 in enumerate(params)
                )
        else:
            ret = tuple(
                tuple(cov.get(iv1,iv2)
                    for iv1,v1 in enumerate(params))\
                    for iv2,v2 in enumerate(params)
                )
        return ret


    def print_matrix(self):
        """show error_matrix"""
        matrix = self.matrix(correlation=True)
        vnames = self.list_of_vary_param()
        self.frontend.print_matrix(vnames, matrix)


    def np_matrix(self, correlation=False):
        """return error/correlation matrix in numpy array format."""
        import numpy as np
        #TODO make a not so lazy one
        return np.array(matrix)


    def is_fixed(self,vname):
        """check if variable *vname* is (initialy) fixed"""
        if vname not in self.parameters:
            raise RuntimeError('Cannot find %s in list of variables.')
        cdef unsigned int index = self.var2pos[vname]
        if self.last_upst is NULL:
            return self.initialfix[vname]
        else:
            return self.last_upst.minuitParameters()[index].isFixed()


    #dealing with frontend conversion
    def print_param(self):
        """print current parameter state"""
        if self.last_upst is NULL:
            self.print_initial_param()
            return
        cdef vector[MinuitParameter] vmps = self.last_upst.minuitParameters()
        cdef int i
        tmp = []
        for i in range(vmps.size()):
            tmp.append(minuitparam2struct(vmps[i]))
        self.frontend.print_param(tmp, self.merrors_struct)


    def print_initial_param(self):
        """Print initial parameters"""
        tmp = []
        for i,vname in enumerate(self.parameters):
            mps = Struct(
            number = i+1,
            name = vname,
            value = self.initialvalue[vname],
            error = self.initialerror[vname],
            is_const = False,
            is_fixed = self.initialfix[vname],
            has_limits = self.initiallimit[vname] is not None,
            lower_limit = self.initiallimit[vname][0]
                if self.initiallimit[vname] is not None else -999,
            upper_limit = self.initiallimit[vname][1]
                if self.initiallimit[vname] is not None else 999,
            has_lower_limit = self.initiallimit[vname] is not None,
            has_upper_limit = self.initiallimit[vname] is not None
            )
            tmp.append(mps)
        self.frontend.print_param(tmp, {})


    def print_fmin(self):
        """print current function minimum state"""
        #cdef MnUserParameterState ust = MnUserParameterState(
        #                               self.cfmin.userState())
        sfmin = cfmin2struct(self.cfmin)
        ncalls = 0 if self.pyfcn is NULL else self.pyfcn.getNumCall()

        self.frontend.print_hline()
        self.frontend.print_fmin(sfmin,self.tol,ncalls)
        self.print_param()
        self.frontend.print_hline()



    def print_all_minos(self):
        """print all minos errors(and its states)"""
        for vname in varnames:
            if vname in self.merrors_struct:
                self.frontend.print_mnerror(vname,self.merrors_struct[vname])


    def set_up(self, double errordef):
        """
        alias for :meth:`set_errordef`
        """
        self.set_errordef(errordef)


    def set_errordef(self, double errordef):
        """
        set error parameter 1 for :math:`\chi^2` and 0.5 for log likelihood

        .. seealso::
            http://wwwasdoc.web.cern.ch/wwwasdoc/minuit/node31.html

        """
        self.errordef = errordef
        if self.pyfcn is not NULL:
            self.pyfcn.set_up(errordef)


    def set_strategy(self,stra):
        """set strategy 0=fast , 1=default, 2=slow but accurate"""
        self.strategy=stra


    def set_print_level(self, lvl):
        """set printlevel 0 quiet, 1 normal, 2 paranoid, 3 really paranoid """
        self.print_level = lvl
        set_migrad_print_level(lvl)


    def get_fmin(self):
        """return current FunctionMinimum Struct"""
        return cfmin2struct(self.cfmin) if self.cfmin is not NULL else None


    #expose internal state using various struct
    def get_param_states(self):
        """Return a list of current MinuitParameter Struct
        for all parameters
        """
        if self.last_upst is NULL:
            return self.get_initial_param_state()
        cdef vector[MinuitParameter] vmps = self.last_upst.minuitParameters()
        cdef int i
        ret = []
        for i in range(vmps.size()):
            ret.append(minuitparam2struct(vmps[i]))
        return ret


    def get_merrors(self):
        """Returns a dictionary of varname-> MinosError Struct"""
        return self.merrors_struct


    def get_initial_param_state(self):
        """get initiail setting inform of MinuitParameter Struct"""
        raise NotImplementedError


    def get_num_call_fcn(self):
        """return number of total call to fcn(not just the last operation)"""
        return 0 if self.pyfcn is NULL else self.pyfcn.getNumCall()


    def migrad_ok(self):
        """check if minimum is valid"""
        return self.cfmin is not NULL and self.cfmin.isValid()


    def matrix_accurate(self):
        """check if covariance(of the last migrad) is accurate."""
        return self.last_upst is not NULL and self.cfmin.hasAccurateCovar()


    def list_of_fixed_param(self):
        """return list of (initially) fixed parameters"""
        return [v for v in self.parameters if self.initialfix[v]]


    def list_of_vary_param(self):
        """return list of (initially) float vary parameters"""
        return [v for v in self.parameters if not self.initialfix[v]]


    #Various utility functions
    cdef construct_FCN(self):
        """(re)construct FCN"""
        del self.pyfcn
        self.pyfcn = new PythonFCN(
                self.fcn,
                self.errordef,
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
                    'Assume 0.') % (vn), InitialParamWarning)
            if 'error_'+vn not in kwds and 'fix_'+param_name(vn) not in kwds:
                warn(('Parameter %s is floating but does not '
                    'have initial step size. Assume 1.') % (vn),
                    InitialParamWarning)
        for vlim in extract_limit(kwds):
            if param_name(vlim) not in self.parameters:
                warn(('%s is given. But there is no parameter %s. '
                    'Ignore.' % (vlim, param_name(vlim)), InitialParamWarning))
        for vfix in extract_fix(kwds):
            if param_name(vfix) not in self.parameters:
                warn(('%s is given. But there is no parameter %s. \
                    Ignore.' % (vfix, param_name(vfix)), InitialParamWarning))
        for verr in extract_error(kwds):
            if param_name(verr) not in self.parameters:
                warn(('%s float. But there is no parameter %s. \
                    Ignore.') % (verr, param_name(verr)), InitialParamWarning)


    def profile(self, vname, bins=100, bound=2, args=None, subtract_min=False):
        """calculate cost function profile around specify range.
        Useful for plotting likelihood scan

        **Arguments**

            * **vname** variable name to scan

            * **bins** number of scanning bin. Default 100.

            * **bound**
                If bound is tuple, (left, right) scanning bound.
                If bound is a number, it specifies how many :math:`\sigma`
                symmetrically from minimum (minimum+- bound* :math:`\sigma`).
                Default 2

            * **subtract_min** subtract_minimum off from return value. This
                makes it easy to label confidence interval. Default False.

        **Returns**

            bins(center point), value
        """
        if isinstance(bound, (int,long,float)):
            start = self.values[vname]
            sigma = self.errors[vname]
            bound = (start+bound*sigma,
                    start-bound*sigma)
        blength = bound[1]-bound[0]
        binstep = blength/(bins-1)
        args = list(self.args) if args is None else args
        #center value
        bins = array.array('d',(bound[0]+binstep*i for i in xrange(bins)))
        ret = array.array('d')
        pos = self.var2pos[vname]
        if subtract_min and self.cfmin is NULL:
            raise RunTimeError("Request for minimization "
                "subtraction but no minimization has been done. "
                "Run migrad first.")
        minval = self.cfmin.fval() if subtract_min else 0.
        for val in bins:
            args[pos] = val
            ret.append(self.fcn(*args)-minval)
        return bins, ret


    def draw_profile(self, vname, bins=100, bound=2, args=None,
        subtract_min=False):
        """
        A convenient wrapper for drawing profile using matplotlib

        .. seealso::

            :meth:`profile`
        """
        return _plotting.draw_profile(self, vname, bins, bound, args,
            subtract_min)

    def contour(self, x, y, bins=20, bound=2, args=None, subtract_min=False):
        """2D countour scan.

        The contour returns is obtained by fixing all others parameters and
        varying **x** and **y**.

        **Arguments**

            - **x** variable name for X axis of scan

            - **y** variable name for Y axis of scan

            - **bound**
                If bound is 2x2 array [[v1min,v1max],[v2min,v2max]].
                If bound is a number, it specifies how many :math:`\sigma`
                symmetrically from minimum (minimum+- bound*:math:`\sigma`).
                Default 2

            - **subtract_min** subtract_minimum off from return value. This
                makes it easy to label confidence interval. Default False.

        **Returns**

            x_bins, y_bins, values

            values[y, x] <-- this choice is so that you can pass it
            to through matplotlib contour()

        .. note::

            If `subtract_min=True`, the return value has the minimum subtracted
            off. The value on the contour can be interpreted *loosely* as
            :math:`i^2 \\times \\textrm{up}` where i is number of standard
            deviation away from the fitted value *WITHOUT* taking into account
            correlation with other parameters that's fixed.

        """
        #don't want to use numpy as requirement for this
        if isinstance(bound, (int,long,float)):
            x_start = self.values[x]
            x_sigma = self.errors[x]
            x_bound = (x_start+bound*x_sigma, x_start-bound*x_sigma)
            y_start = self.values[y]
            y_sigma = self.errors[y]
            y_bound = (y_start+bound*y_sigma, y_start-bound*y_sigma)
        else:
            x_bound = bound[0]
            y_bound = bound[1]

        x_bins = bins
        y_bins = bins

        x_blength = x_bound[1]-x_bound[0]
        x_binstep = x_blength/(x_bins-1)

        y_blength = y_bound[1]-y_bound[0]
        y_binstep = y_blength/(y_bins-1)

        x_val = array.array('d',(x_bound[0]+x_binstep*i for i in xrange(x_bins)))
        y_val = array.array('d',(y_bound[0]+y_binstep*i for i in xrange(y_bins)))

        x_pos = self.var2pos[x]
        y_pos = self.var2pos[y]

        args = list(self.args) if args is None else args

        if subtract_min and self.cfmin is NULL:
            raise RunTimeError("Request for minimization "
                "subtraction but no minimization has been done. "
                "Run migrad first.")
        minval = self.cfmin.fval() if subtract_min else 0.

        ret = list()
        for yy in y_val:
            args[y_pos] = yy
            tmp = array.array('d')
            for xx in x_val:
                args[x_pos] = xx
                tmp.append(self.fcn(*args)-minval)
            ret.append(tmp)

        return x_val, y_val, ret


    def draw_contour(self, x, y, bins=20, bound=2, args=None, show_sigma=True):
        """
        Convenient wrapper for drawing contour. The argument is the same as
        :meth:`contour`. If `show_sigma=True`(Default), the label on the
        contour lines will show how many :math:`\sigma` away from the optimal
        value instead of raw value.

        .. note::

            Like :meth:`contour`. The error shown on the plot is not strictly
            1 :math:`\sigma` contour since the other parameters are fixed.

        .. seealso::

            :meth:`contour`

        """
        return _plotting.draw_contour(self, x, y, bins, bound, args, show_sigma)

    cdef refreshInternalState(self):
        """refresh internal state attributes.
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
                self.errors[mpv[i].name()] = mpv[i].error()
            self.args = tuple(self.args)
            self.fitarg.update(self.values)
            self.fitarg.update({'error_'+k:v for k,v in self.errors.items()})
            vary_param = self.list_of_vary_param()
            if self.last_upst.hasCovariance():
                cov = self.last_upst.covariance()
                self.covariance =\
                     {(v1,v2):cov.get(i,j)\
                         for i,v1 in enumerate(vary_param)\
                         for j,v2 in enumerate(vary_param)}
            else:
                self.covariance = None
            self.fval = self.last_upst.fval()
            self.ncalls = self.last_upst.nfcn()
            self.edm = self.last_upst.edm()
            self.gcc = None
            if self.last_upst.hasGlobalCC() and self.last_upst.globalCC().isValid():
                self.gcc = {v:self.last_upst.globalCC().globalCC().at(i)\
                    for i,v in enumerate(self.list_of_vary_param())}
        self.merrors = {(k,1.0):v.upper
                       for k,v in self.merrors_struct.items()}
        self.merrors.update({(k,-1.0):v.lower
                       for k,v in self.merrors_struct.items()})


    cdef MnUserParameterState* initialParameterState(self):
        """construct parameter state from initial array.
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
                if lb >= ub:
                    raise ValueError('limit for parameter %s is invalid. %r'(v,(lb,ub)))
                ret.setLimits(v,lb,ub)
        for v in self.parameters:
            if self.initialfix[v]:
                ret.fix(v)
        return ret


    def _auto_frontend(self):
        """determine front end automatically.
        If this session is an IPYTHON session then use Html frontend,
        Console Frontend otherwise.
        """
        try:
            __IPYTHON__
            from HtmlFrontend import HtmlFrontend
            return HtmlFrontend()
        except NameError:
            return ConsoleFrontend()


    def _check_extra_args(self,parameters,kwd):
        """check keyword arguments to find unwanted/typo keyword arguments"""
        fixed_param = set('fix_'+p for p in parameters)
        limit_param = set('limit_'+p for p in parameters)
        error_param = set('error_'+p for p in parameters)
        for k in kwd.keys():
            if k not in parameters and\
                    k not in fixed_param and\
                    k not in limit_param and\
                    k not in error_param:
                raise RuntimeError(
                        ('Cannot understand keyword %s. May be a typo?\n'
                        'The parameters are %r')%(k,parameters))
