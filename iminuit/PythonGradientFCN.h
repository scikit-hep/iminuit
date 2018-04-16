#ifndef IMINUIT_PYTHONGRADIENTFCN_H
#define IMINUIT_PYTHONGRADIENTFCN_H

#include <vector>
#include <string>
#include <Python.h>
#include "Minuit2/FCNGradientBase.h"
#include "IMinuitMixin.h"
#include "PythonCaller.h"

using namespace ROOT::Minuit2;

class PythonGradientFCN :
    public FCNGradientBase,
    public IMinuitMixin {
public:
    PythonGradientFCN() {} //for cython stack allocate

    PythonGradientFCN(PyObject* fcn,
        PyObject* gradfcn,
        double up,
        const std::vector<std::string>& names,
        bool thrownan) :
        IMinuitMixin(up, names, thrownan),
        call_fcn(fcn),
        call_grad(gradfcn)
    {}

    virtual ~PythonGradientFCN() {}

    virtual double operator()(const std::vector<double>& x) const{
        return call_fcn.scalar<vector2tuple>(x, names, throw_nan);
    }

    virtual std::vector<double> Gradient(const std::vector<double>& x) const{
        return call_grad.vector<vector2tuple>(x, names, throw_nan);
    }

    virtual double Up() const { return up; }
    virtual void SetErrorDef(double x) { up = x; }

    virtual int getNumCall() const { return call_fcn.ncall; }
    virtual void resetNumCall() { call_fcn.ncall = 0; }

private:
    PythonCaller call_fcn, call_grad;
};

class NumpyGradientFCN :
    public FCNGradientBase,
    public IMinuitMixin {
public:
    NumpyGradientFCN() {} //for cython stack allocate

    NumpyGradientFCN(PyObject* fcn,
        PyObject* gradfcn,
        double up,
        const std::vector<std::string>& names,
        bool thrownan) :
        IMinuitMixin(up, names, thrownan),
        call_fcn(fcn),
        call_grad(gradfcn)
    {}

    virtual ~NumpyGradientFCN() {}

    virtual double operator()(const std::vector<double>& x) const{
        return call_fcn.scalar<vector2array>(x, names, throw_nan);
    }

    virtual std::vector<double> Gradient(const std::vector<double>& x) const{
        return call_grad.vector<vector2array>(x, names, throw_nan);
    }

    virtual double Up() const { return up; }
    virtual void SetErrorDef(double x) { up = x; }

    virtual int getNumCall() const { return call_fcn.ncall; }
    virtual void resetNumCall() { call_fcn.ncall = 0; }

private:
    PythonCaller call_fcn, call_grad;
};

#endif
