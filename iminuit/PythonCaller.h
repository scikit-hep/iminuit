#ifndef IMINUIT_PYTHONCALLER_H
#define IMINUIT_PYTHONCALLER_H

#include <stdexcept>
#include <string>
#include <vector>
#include <Python.h>
// cannot use this, because cython enforces old api:
// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include <cmath>
#include "Utils.h"

namespace detail {

//warn but do not reset the error flag
inline void warn_preserve_error(const std::string& msg) {
    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    PyErr_Warn(NULL, msg.c_str());
    PyErr_Restore(ptype, pvalue, ptraceback);
}

inline std::string errormsg(const char* prefix,
                            const std::vector<std::string>& pname,
                            const std::vector<double>& x) {
    std::string ret = prefix;
    ret += "fcn is called with following arguments:\n";
    assert(pname.size() == x.size());
    //determine longest variable length
    size_t maxlength = 0;
    for (int i=0; i < x.size(); i++) {
        maxlength = std::max(pname[i].size(),
                             maxlength);
    }
    for (int i=0; i < x.size(); i++) {
        std::string line = format("%*s = %+f\n",
                                  maxlength+4,
                                  pname[i].c_str(),
                                  x[i]);
        ret += line;
    }
    return ret;
}

inline bool isnan(double x) {
    #if defined(_WIN64)
        return _isnanf(x);
    #elif defined(_WIN32)
        return _isnan(x);
    #else
        return std::isnan(x);
    #endif
}

} // namespace detail

// typedef for function ptr
typedef PyObject* (*ConvertFunction)(const std::vector<double>&);

PyObject* vector2tuple(const std::vector<double>& x) {
    PyObject* tuple = PyTuple_New(x.size());
    for (int i = 0; i < x.size(); ++i) {
        PyObject* data = PyFloat_FromDouble(x[i]);
        PyTuple_SET_ITEM(tuple, i, data); // steals ref
    }
    return tuple; //new reference
}

PyObject* vector2array(const std::vector<double>& x) {
    npy_intp dims = x.size();
    PyObject* seq = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE,
                                              const_cast<double*>(&x[0]));
    PyObject* tuple = PyTuple_New(1);
    PyTuple_SET_ITEM(tuple, 0, seq); // steals ref
    return tuple; //new reference
}


/**
    This class encapsulates the callback to a Python function from C++.
    It is responsible for:
    - Handling the Refcount of the Python function
    - Converting from native C++ types to Python types and back
    - Keeping a function call counter
    - Handling exceptions on the Python side and converting those to C++ exceptions

    Two kinds of a Python function signatures are supported by the functions
    vector2tuple and vector2array. The first converts a C++ vector of doubles
    to a Python tuple of doubles. The second converts the C++ vector to a
    Numpy array view of the C++ vector and puts that array into a Python tuple.
*/
class PythonCaller {
public:
    PythonCaller() : fcn(NULL), convert(NULL), ncall(0) {}

    PythonCaller(const PythonCaller& x) :
        fcn(x.fcn), convert(x.convert), ncall(x.ncall)
    {
        Py_INCREF(fcn);
    }

    PythonCaller(PyObject* pfcn, bool use_array_call) :
        fcn(pfcn), convert(use_array_call ? vector2array : vector2tuple),
        ncall(0)
    {
        Py_INCREF(fcn);
    }

    ~PythonCaller() {
        Py_DECREF(fcn);
    }

    double scalar(const std::vector<double>& x,
                  const std::vector<std::string>& names,
                  const bool throw_nan) const {
        PyObject* args = convert(x); // no error can occur here
        PyObject* result = PyObject_Call(fcn, args, NULL);
        Py_DECREF(args);

        if (PyErr_Occurred()) {
            std::string msg = detail::errormsg("Exception Occured\n",
                                               names, x);
            detail::warn_preserve_error(msg.c_str());
            Py_XDECREF(result);
            throw std::runtime_error(msg);
        }

        const double ret = PyFloat_AsDouble(result);
        Py_DECREF(result);

        if (PyErr_Occurred()) {
            std::string msg = detail::errormsg("Cannot convert call result to double\n",
                                       names, x);
            detail::warn_preserve_error(msg.c_str());
            throw std::runtime_error(msg);
        }

        if (detail::isnan(ret)) {
            std::string msg = detail::errormsg("result is Nan\n",
                                       names, x);
            detail::warn_preserve_error(msg.c_str());
            if (throw_nan) {
                PyErr_SetString(PyExc_RuntimeError, msg.c_str());
                throw std::runtime_error(msg.c_str());
            }
        }

        ++ncall;
        return ret;
    }

    std::vector<double> vector(const std::vector<double>& x,
                               const std::vector<std::string>& names,
                               const bool throw_nan) const {
        PyObject* args = convert(x); // no error can occur here
        PyObject* result = PyObject_Call(fcn, args, NULL);
        Py_DECREF(args);

        if (PyErr_Occurred()) {
            std::string msg = detail::errormsg("exception occured\n",
                                       names, x);
            detail::warn_preserve_error(msg.c_str());
            Py_XDECREF(result);
            throw std::runtime_error(msg);
        }

        // Convert the iterable to a vector
        PyObject *iterator = PyObject_GetIter(result);
        if (iterator == NULL) {
            std::string msg = detail::errormsg("result must be iterable\n",
                                               names, x);
            detail::warn_preserve_error(msg.c_str());
            Py_XDECREF(result);
            throw std::runtime_error(msg);
        }

        std::vector<double> result_vector;
        result_vector.reserve(PySequence_Size(result));
        PyObject *item = NULL;
        while (item = PyIter_Next(iterator)) {
            const double xi = PyFloat_AsDouble(item);
            Py_DECREF(item);

            if (PyErr_Occurred()) {
                std::string msg = detail::errormsg("cannot convert to vector of doubles\n",
                                                   names, x);
                detail::warn_preserve_error(msg.c_str());
                throw std::runtime_error(msg);
            }

            if (detail::isnan(xi)) {
                std::string msg = detail::errormsg("result is NaN\n",
                                                   names, x);
                detail::warn_preserve_error(msg.c_str());
                if (throw_nan) {
                    PyErr_SetString(PyExc_RuntimeError, msg.c_str());
                    Py_DECREF(iterator);
                    throw std::runtime_error(msg.c_str());
                }
            }
            result_vector.push_back(xi);
        }

        Py_DECREF(iterator);
        Py_DECREF(result);

        ++ncall;
        return result_vector;
    }

private:
    PyObject* fcn;
    ConvertFunction convert;

public:
    mutable int ncall;
};

#endif
