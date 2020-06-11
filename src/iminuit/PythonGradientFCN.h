#ifndef IMINUIT_PYTHONGRADIENTFCN_H
#define IMINUIT_PYTHONGRADIENTFCN_H

#include <vector>
#include <string>
#include <Python.h>
#include "Minuit2/FCNGradientBase.h"
#include "IMinuitMixin.h"
#include "PythonCaller.h"

using namespace ROOT::Minuit2;

/**
    Called by Minuit2 to call the underlying Python function which computes
    the objective function and its gradient. The interface of this class is
    defined by the abstract base class FCNGradientBase.

    This version calls the function with a tuple of numbers or a single Numpy
    array, depending on the ConvertFunction argument.
*/
class PythonGradientFCN :
    public FCNGradientBase,
    public IMinuitMixin {
public:
    PythonGradientFCN() {} //for cython stack allocate

    PythonGradientFCN(PyObject* fcn,
        PyObject* gradfcn,
        bool use_array_call,
        double up,
        const std::vector<std::string>& names,
        bool thrownan) :
        IMinuitMixin(up, names, thrownan),
        call_fcn(fcn, use_array_call),
        call_grad(gradfcn, use_array_call)
    {}

    virtual ~PythonGradientFCN() {}

    virtual double operator()(const std::vector<double>& x) const{
        return call_fcn.scalar(x, names, throw_nan);
    }

    virtual std::vector<double> Gradient(const std::vector<double>& x) const{
        return call_grad.vector(x, names, throw_nan);
    }

    virtual double Up() const { return up; }
    virtual void SetErrorDef(double x) { up = x; }

    virtual int getNumCall() const { return call_fcn.ncall; }
    virtual void resetNumCall() { call_fcn.ncall = 0; }

    virtual int getNumGrad() const { return call_grad.ncall; }
    virtual void resetNumGrad() { call_grad.ncall = 0; }

    // prevent Minuit2 from computing gradients numerically to check analytical gradient
    virtual bool CheckGradient() const {return false;}

private:
    PythonCaller call_fcn, call_grad;
};

#endif
