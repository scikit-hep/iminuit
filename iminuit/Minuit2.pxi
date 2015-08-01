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
        double call "operator()"(vector[double] x) except+
        double ErrorDef()

cdef extern from "Minuit2/MnApplication.h":
    cdef cppclass MnApplication:
        FunctionMinimum call "operator()"(int, double) except+
        void SetPrecision(double)

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

cdef extern from "Minuit2/MnStrategy.h":
    cdef cppclass MnStrategy:
        MnStrategy(unsigned int)

cdef extern from "Minuit2/MnMigrad.h":
    cdef cppclass MnMigrad(MnApplication):
        MnMigrad(FCNBase fcn, MnUserParameterState par, MnStrategy str) except+
        FunctionMinimum call "operator()"(int, double) except+
        void SetPrecision(double)

cdef extern from "Minuit2/MnHesse.h":
    cdef cppclass MnHesse:
        MnHesse(unsigned int stra)
        MnUserParameterState call "operator()"(FCNBase, MnUserParameterState, unsigned int maxcalls=0) except+

cdef extern from "Minuit2/MnMinos.h":
    cdef cppclass MnMinos:
        MnMinos(FCNBase fcn, FunctionMinimum min, unsigned int stra)
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

cdef extern from "Minuit2/VariableMetricBuilder.h":
    void set_migrad_print_level "VariableMetricBuilder::setPrintLevel"(int p)

cdef extern from "Minuit2/MnContours.h":
    cdef cppclass MnContours:
        MnContours(FCNBase fcn, FunctionMinimum fm, unsigned int stra)
        ContoursError Contour(unsigned int, unsigned int, unsigned int npoints)

cdef extern from "Minuit2/ContoursError.h":
    cdef cppclass ContoursError:
        ContoursError()
        vector[pair[double, double]] Points "operator()"()
        MinosError XMinosError()
        MinosError YMinosError()
