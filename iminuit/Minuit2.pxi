"""Minuit C++ class interface.
"""
from libcpp.vector cimport vector
from libcpp.utility cimport pair

cdef extern from "<memory>" namespace "std":
    cdef cppclass auto_ptr[T]:
        auto_ptr()
        auto_ptr(T*ptr)
        T*get()

cdef extern from "Minuit2/FCNBase.h":
    cdef cppclass FCNBase:
        double call "operator()"(vector[double] x) except +
        double Up()
        void SetErrorDef(double)

cdef extern from "Minuit2/FCNGradientBase.h":
    cdef cppclass FCNGradientBase(FCNBase):
        vector[double] call "Gradient"(vector[double] x) except +

cdef extern from "IMinuitMixin.h":
    cdef cppclass IMinuitMixin:
        int getNumCall()
        void resetNumCall()

cdef extern from "Utils.h":
    FunctionMinimum* call_mnapplication_wrapper(
        MnApplication app, unsigned int i, double tol) except +

cdef extern from "PythonFCN.h":
    cdef cppclass PythonFCN(FCNBase, IMinuitMixin):
        PythonFCN(object fcn, bint use_array_call, double up_parm, vector[string] pname, bint thrownan)

cdef extern from "PythonGradientFCN.h":
    cdef cppclass PythonGradientFCN(FCNGradientBase, IMinuitMixin):
        PythonGradientFCN(object fcn, object grad, bint use_array_call, double up_parm, vector[string] pname, bint thrownan)
        int getNumGrad()
        void resetNumGrad()

cdef extern from "Minuit2/FunctionMinimum.h":
    cdef cppclass FunctionMinimum:
        bint IsValid()
        MnUserParameterState UserState()
        bint HasAccurateCovar()
        double Fval()
        double Edm()
        int NFcn()
        double Up()
        bint HasValidParameters()
        bint HasPosDefCovar()
        bint HasMadePosDefCovar()
        bint HesseFailed()
        bint HasCovariance()
        bint HasReachedCallLimit()
        bint IsAboveMaxEdm()

cdef extern from "Minuit2/MinimumBuilder.h":
    cdef cppclass MinimumBuilder:
        int StorageLevel()
        int PrintLevel()
        void SetPrintLevel(int)
        void SetStorageLevel(int)

cdef extern from "Minuit2/ModularFunctionMinimizer.h":
    cdef cppclass ModularFunctionMinimizer:
        MinimumBuilder Builder()

cdef extern from "Minuit2/MnApplication.h":
    cdef cppclass MnApplication:
        FunctionMinimum call "operator()"(int, double) except+
        void SetPrecision(double)
        ModularFunctionMinimizer Minimizer()

cdef extern from "Minuit2/MinuitParameter.h":
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

cdef extern from "Minuit2/MnUserCovariance.h":
    cdef cppclass MnUserCovariance:
        unsigned int Nrow()
        double get "operator()"(unsigned int row, unsigned int col)

cdef extern from "Minuit2/MnGlobalCorrelationCoeff.h":
    cdef cppclass MnGlobalCorrelationCoeff:
        vector[double] GlobalCC()
        bint IsValid()

cdef extern from "Minuit2/MnUserParameterState.h":
    cdef cppclass MnUserParameterState:
        MnUserParameterState()
        MnUserParameterState(MnUserParameterState mpst)
        operator=(MnUserParameterState)

        void Add(char*name, double val, double err)
        void Add(char*name, double val, double err, double, double)
        void Add(char*, double)

        vector[MinuitParameter] MinuitParameters()
        MnUserCovariance Covariance()
        MnGlobalCorrelationCoeff GlobalCC()

        double Fval()
        double Edm()
        unsigned int NFcn()

        const MinuitParameter& Parameter(unsigned int)

        void Fix(unsigned int)
        void Release(unsigned int)
        void SetValue(unsigned int, double)
        void SetError(unsigned int, double)
        void SetLimits(unsigned int, double, double)
        void SetUpperLimit(unsigned int, double)
        void SetLowerLimit(unsigned int, double)
        void RemoveLimits(unsigned int)

        bint IsValid()
        bint HasCovariance()
        bint HasGlobalCC()

        unsigned int Index(char*)
        char*Name(unsigned int)

cdef extern from "Minuit2/MnStrategy.h":
    cdef cppclass MnStrategy:
        MnStrategy(unsigned int)

cdef extern from "Minuit2/MnMigrad.h":
    cdef cppclass MnMigrad(MnApplication):
        MnMigrad(FCNBase fcn, MnUserParameterState par, MnStrategy str) except+
        MnMigrad(FCNGradientBase fcn, MnUserParameterState par, MnStrategy str) except+
        FunctionMinimum call "operator()"(int, double) except+
        void SetPrecision(double)

cdef extern from "Minuit2/MnHesse.h":
    cdef cppclass MnHesse:
        MnHesse(unsigned int stra)
        MnUserParameterState call "operator()"(FCNBase, MnUserParameterState, unsigned int maxcalls) except+
        MnUserParameterState call "operator()"(FCNGradientBase, MnUserParameterState, unsigned int maxcalls) except+

cdef extern from "Minuit2/MnMinos.h":
    cdef cppclass MnMinos:
        MnMinos(FCNBase fcn, FunctionMinimum min, unsigned int stra)
        MnMinos(FCNGradientBase fcn, FunctionMinimum min, unsigned int stra)
        MinosError Minos(unsigned int par, unsigned int maxcalls) except +

cdef extern from "Minuit2/MinosError.h":
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

cdef extern from "Minuit2/FunctionMinimum.h":
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

cdef extern from "Minuit2/MnContours.h":
    cdef cppclass MnContours:
        MnContours(FCNBase fcn, FunctionMinimum fm, unsigned int stra)
        MnContours(FCNGradientBase fcn, FunctionMinimum fm, unsigned int stra)
        ContoursError Contour(unsigned int, unsigned int, unsigned int npoints)

cdef extern from "Minuit2/ContoursError.h":
    cdef cppclass ContoursError:
        ContoursError()
        vector[pair[double, double]] Points "operator()"()
        MinosError XMinosError()
        MinosError YMinosError()
