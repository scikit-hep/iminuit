"""Minuit C++ class interface.
"""
from libcpp.vector cimport vector
from libcpp.utility cimport pair

cdef extern from "<memory>" namespace "std":
    cdef cppclass unique_ptr[T]:
        unique_ptr()
        unique_ptr(T*ptr)
        T*get()

cdef extern from "Minuit2/FCNBase.h" namespace "ROOT::Minuit2":
    cdef cppclass FCNBase:
        double call "operator()"(vector[double] x) except+
        double ErrorDef()

cdef extern from "Minuit2/FCNGradientBase.h" namespace "ROOT::Minuit2":
    cdef cppclass FCNGradientBase:
        FCNGradientBase(object fcn, double up_parm, vector[string] pname, bint thrownan)
        double call "operator()"(vector[double] x) except +  #raise_py_err
        double Up()
        vector[double] Gradient(vector[double] x) except +  #raise_py_err
        bint CheckGradient()

cdef extern from "Minuit2/MinimumBuilder.h" namespace "ROOT::MinimumBuilder":
    cdef cppclass MinimumBuilder:
        int StorageLevel()
        int PrintLevel()
        void SetPrintLevel(int)
        void SetStorageLevel(int)

cdef extern from "Minuit2/ModularFunctionMinimizer.h" namespace "ROOT::Minuit2":
    cdef cppclass ModularFunctionMinimizer:
        MinimumBuilder Builder()

cdef extern from "Minuit2/MnApplication.h" namespace "ROOT::Minuit2":
    cdef cppclass MnApplication:
        FunctionMinimum call "operator()"(int, double) except+
        void SetPrecision(double)
        ModularFunctionMinimizer Minimizer()

cdef extern from "Minuit2/MinuitParameter.h" namespace "ROOT::Minuit2":
    cdef cppclass MinuitParameter:
        MinuitParameter(unsigned int, char*, double)
        unsigned int Number()
        char*Name()
        double Value()
        double Error()
        bint IsConst()
        bint IsFixed()

        bint HasLimits()
        bint HasLowerLimit()
        bint HasUpperLimit()
        double LowerLimit()
        double UpperLimit()

cdef extern from "Minuit2/MnUserCovariance.h" namespace "ROOT::Minuit2":
    cdef cppclass MnUserCovariance:
        unsigned int Nrow()
        double get "operator()"(unsigned int row, unsigned int col)

cdef extern from "Minuit2/MnGlobalCorrelationCoeff.h" namespace "ROOT::Minuit2":
    cdef cppclass MnGlobalCorrelationCoeff:
        vector[double] GlobalCC()
        bint IsValid()

cdef extern from "Minuit2/MnUserParameterState.h" namespace "ROOT::Minuit2":
    cdef cppclass MnUserParameterState:
        MnUserParameterState()
        MnUserParameterState(MnUserParameterState mpst)
        vector[double] Params()
        void Add(char*name, double val, double err)
        void Add(char*name, double val, double err, double, double)
        void Add(char*, double)

        vector[MinuitParameter] MinuitParameters()
        #MnUserParameters parameters()
        MnUserCovariance Covariance()
        MnGlobalCorrelationCoeff GlobalCC()

        double Fval()
        double Edm()
        unsigned int NFcn()

        void Fix(char*)
        void Release(char*)
        void SetValue(char*, double)
        void SetError(char*, double)
        void SetLimits(char*, double, double)
        void SetUpperLimit(char*, double)
        void SetLowerLimit(char*, double)
        void RemoveLimits(char*)

        bint IsValid()
        bint HasCovariance()
        bint HasGlobalCC()

        double Value(char*)
        double Error(char*)

        unsigned int Index(char*)
        char*Name(unsigned int)

cdef extern from "Minuit2/MnStrategy.h" namespace "ROOT::Minuit2":
    cdef cppclass MnStrategy:
        MnStrategy(unsigned int)

cdef extern from "Minuit2/MnMigrad.h" namespace "ROOT::Minuit2":
    cdef cppclass MnMigrad(MnApplication):
        MnMigrad(FCNBase fcn, MnUserParameterState par, MnStrategy str) except+
        MnMigrad(FCNGradientBase fcn, MnUserParameterState par, MnStrategy str) except+
        FunctionMinimum call "operator()"(int, double) except+

cdef extern from "Minuit2/MnHesse.h" namespace "ROOT::Minuit2":
    cdef cppclass MnHesse:
        MnHesse(unsigned int stra)
        MnUserParameterState call "operator()"(FCNBase, MnUserParameterState, unsigned int maxcalls=0) except+
        MnUserParameterState call "operator()"(FCNGradientBase, MnUserParameterState, unsigned int maxcalls=0) except+

cdef extern from "Minuit2/MnMinos.h" namespace "ROOT::Minuit2":
    cdef cppclass MnMinos:
        MnMinos(FCNBase fcn, FunctionMinimum min, unsigned int stra)
        MnMinos(FCNGradientBase fcn, FunctionMinimum min, unsigned int stra)
        MinosError Minos(unsigned int par, unsigned int maxcalls) except +

cdef extern from "Minuit2/MinosError.h" namespace "ROOT::Minuit2":
    cdef cppclass MinosError:
        double Lower()
        double Upper()
        bint IsValid()
        bint LowerValid()
        bint UpperValid()
        bint AtLowerLimit()
        bint AtUpperLimit()
        bint AtLowerMaxFcn()
        bint AtUpperMaxFcn()
        bint LowerNewMin()
        bint UpperNewMin()
        unsigned int NFcn()
        double Min()

cdef extern from "Minuit2/FunctionMinimum.h" namespace "ROOT::Minuit2":
    cdef cppclass FunctionMinimum:
        FunctionMinimum(FunctionMinimum)
        MnUserParameterState UserState()
        MnUserCovariance UserCovariance()
        # const_MinimumParameter parameters()
        # const_MinimumError error()

        double Fval()
        double Edm()
        int NFcn()

        double Up()
        bint HasValidParameters()
        bint IsValid()
        bint HasValidCovariance()
        bint HasAccurateCovar()
        bint HasPosDefCovar()
        bint HasMadePosDefCovar()
        bint HesseFailed()
        bint HasCovariance()
        bint HasReachedCallLimit()
        bint IsAboveMaxEdm()

cdef extern from "Minuit2/MnContours.h" namespace "ROOT::Minuit2":
    cdef cppclass MnContours:
        MnContours(FCNBase fcn, FunctionMinimum fm, unsigned int stra)
        MnContours(FCNGradientBase fcn, FunctionMinimum fm, unsigned int stra)
        ContoursError Contour(unsigned int, unsigned int, unsigned int npoints)

cdef extern from "Minuit2/ContoursError.h" namespace "ROOT::Minuit2":
    cdef cppclass ContoursError:
        ContoursError()
        vector[pair[double, double]] Points "operator()"()
        MinosError XMinosError()
        MinosError YMinosError()
