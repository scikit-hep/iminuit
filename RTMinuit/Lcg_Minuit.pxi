from libcpp.vector cimport vector
from libcpp cimport bool


#LCG Minuit
cdef extern from "Minuit/FCNBase.h":
    cdef cppclass FCNBase:
        double call "operator()" (vector[double] x) except+
        double errorDef()
        double up()

cdef extern from "Minuit/MnApplication.h":
    cdef cppclass MnApplication:
         FunctionMinimum call "operator()" (int,double) except+

cdef extern from "Minuit/MinuitParameter.h":
    cdef cppclass MinuitParameter:
        MinuitParameter(unsigned int, char*, double)
        unsigned int number()
        char* name()
        double value()
        double error()
        bint isConst()
        bint isFixed()

        bint hasLimits()
        bint hasLowerLimit()
        bint hasUpperLimit()
        double lowerLimit()
        double upperLimit()

cdef extern from "Minuit/MnUserCovariance.h":
    cdef cppclass MnUserCovariance:
        double get "operator()" (unsigned int row, unsigned int col)

cdef extern from "Minuit/MnGlobalCorrelationCoeff.h":
    cdef cppclass MnGlobalCorrelationCoeff:
        vector[double] globalCC()

cdef extern from "Minuit/MnUserParameterState.h":
    cdef cppclass MnUserParameterState:
        MnUserParameterState()
        MnUserParameterState(MnUserParameterState mpst)
        vector[double] params
        void add(char* name, double val, double err)
        void add(char* name, double val, double err, double , double)
        void add(char*, double)

        vector[MinuitParameter] minuitParameters()
        #MnUserParameters parameters()
        MnUserCovariance covariance()
        MnGlobalCorrelationCoeff globalCC()

        void fix(char*)
        void release(char*)
        void setValue(char*, double)
        void setError(char*, double)
        void setLimits(char*, double, double)
        void setUpperLimit(char*, double)
        void setLowerLimit(char*, double)
        void removeLimits(char*)

        bint isValid()
        bint hasCovariance()
        bint hasGlobalCC()

        double value(char*)
        double error(char*)

        unsigned int index(char*)
        char* name(unsigned int)

cdef extern from "Minuit/MnStrategy.h":
    cdef cppclass MnStrategy:
        MnStrategy(unsigned int)


cdef extern from "Minuit/MnMigrad.h":
    cdef cppclass MnMigrad(MnApplication):
        MnMigrad(FCNBase fcn, MnUserParameterState par, MnStrategy str ) except+
        FunctionMinimum call "operator()" (int,double) except+

cdef extern from "Minuit/FunctionMinimum.h":
    cdef cppclass FunctionMinimum:
        FunctionMinimum(FunctionMinimum)
        MnUserParameterState userState()
        MnUserCovariance userCovariance()
        
        # const_MinimumParameter parameters()
        # const_MinimumError error()

        double fval()
        double edm()
        int nfcn()

        double up()
        bint hasValidParameters()
        bint isValid()
        bint hasValidCovariance()
        bint hasAccurateCovar()
        bint hasPosDefCovar()
        bint hasMadePosDefCovar()
        bint hesseFailed()
        bint hasCovariance()
        bint hasReachedCallLimit()
        bint isAboveMaxEdm()
