#ifndef IMINUIT_PYTHONFCN_H
#define IMINUIT_PYTHONFCN_H

#include <vector>
#include <string>
#include <Python.h>
#include "Minuit2/FCNBase.h"
#include "IMinuitMixin.h"
#include "PythonCaller.h"

using namespace ROOT::Minuit2;

class PythonFCN :
    public FCNBase,
    public IMinuitMixin {
public:
    PythonFCN() {} //for cython stack allocate

    PythonFCN(PyObject* fcn,
              double up,
              const std::vector<std::string>& pname,
              bool thrownan = false) :
        IMinuitMixin(up, pname, thrownan),
        call_fcn(fcn)
    {}

    PythonFCN(const PythonFCN& x) {}

    virtual ~PythonFCN() {}

    virtual double operator()(const std::vector<double>& x) const{
        return call_fcn.scalar<vector2tuple>(x, names, throw_nan);
    }

    virtual double Up() const { return up; }
    virtual void SetErrorDef(double x) { up = x; }

    virtual int getNumCall() const { return call_fcn.ncall; }
    virtual void resetNumCall() { call_fcn.ncall = 0; }

private:
    PythonCaller call_fcn;
};

class NumpyFCN :
    public FCNBase,
    public IMinuitMixin {
public:
    NumpyFCN() {} //for cython stack allocate

    NumpyFCN(PyObject* fcn,
              double up,
              const std::vector<std::string>& pname,
              bool thrownan = false) :
        IMinuitMixin(up, pname, thrownan),
        call_fcn(fcn)
    {}

    virtual ~NumpyFCN() {}

    virtual double operator()(const std::vector<double>& x) const{
        return call_fcn.scalar<vector2array>(x, names, throw_nan);
    }

    virtual double Up() const { return up; }
    virtual void SetErrorDef(double x) { up = x; }

    virtual int getNumCall() const { return call_fcn.ncall; }
    virtual void resetNumCall() { call_fcn.ncall = 0; }

private:
    PythonCaller call_fcn;
};
#endif
