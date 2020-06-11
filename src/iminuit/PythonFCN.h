#ifndef IMINUIT_PYTHONFCN_H
#define IMINUIT_PYTHONFCN_H

#include <vector>
#include <string>
#include <Python.h>
#include "Minuit2/FCNBase.h"
#include "IMinuitMixin.h"
#include "PythonCaller.h"

using namespace ROOT::Minuit2;

/**
    Called by Minuit2 to call the underlying Python function which computes
    the objective function. The interface of this class is defined by the
    abstract base class FCNBase.

    This version calls the function with a tuple of numbers or a single Numpy
    array, depending on the ConvertFunction argument.
*/
class PythonFCN :
    public FCNBase,
    public IMinuitMixin {
public:
    PythonFCN() {} //for cython stack allocate

    PythonFCN(PyObject* fcn,
              bool use_array_call,
              double up,
              const std::vector<std::string>& pname,
              bool thrownan = false) :
        IMinuitMixin(up, pname, thrownan),
        call_fcn(fcn, use_array_call)
    {}

    PythonFCN(const PythonFCN& x) {}

    virtual ~PythonFCN() {}

    virtual double operator()(const std::vector<double>& x) const{
        return call_fcn.scalar(x, names, throw_nan);
    }

    virtual double Up() const { return up; }
    virtual void SetErrorDef(double x) { up = x; }

    virtual int getNumCall() const { return call_fcn.ncall; }
    virtual void resetNumCall() { call_fcn.ncall = 0; }

private:
    PythonCaller call_fcn;
};

#endif
