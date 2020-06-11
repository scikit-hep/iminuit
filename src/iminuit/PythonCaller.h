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

inline std::string errormsg(const char* prefix,
                            const std::vector<std::string>& pname,
                            const std::vector<double>& x) {
    std::string ret;
    ret += prefix;
    ret += "\nUser function arguments:\n";
    assert(pname.size() == x.size());
    //determine longest variable length
    size_t maxlength = 0;
    for (std::size_t i=0; i < x.size(); i++) {
        maxlength = std::max(pname[i].size(),
                             maxlength);
    }
    for (std::size_t i=0; i < x.size(); i++) {
        std::string line = format("%*s = %+f\n",
                                  maxlength+4,
                                  pname[i].c_str(),
                                  x[i]);
        ret += line;
    }

    PyHandle ptype, pvalue, ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);

    if (ptype && pvalue) {
        // add original Python error report
        PyHandle util_module = PyImport_ImportModule("iminuit.util");
        if (!util_module) std::abort(); // should never happen

        PyHandle format = PyObject_GetAttrString(util_module, "format_exception");
        if (!(format && PyCallable_Check(format))) std::abort(); // should never happen

        PyHandle s;
        s = PyObject_CallFunctionObjArgs(format, (PyObject*)ptype,
                                         (PyObject*)pvalue,
                                         ptraceback ? (PyObject*)ptraceback : Py_None, NULL);

        if (s) {
            ret += "Original python exception in user function:\n";
#if PY_MAJOR_VERSION < 3
            ret += PyString_AsString(s);
#else
            PyHandle b = PyUnicode_AsEncodedString(s, "ascii", "xmlcharrefreplace");
            ret += PyBytes_AsString(b);
#endif
        }
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
    for (std::size_t i = 0; i < x.size(); ++i) {
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
        PyHandle args, result;
        args = convert(x); // no error can occur here
        result = PyObject_CallObject(fcn, args);

        if (!result) {
            std::string msg = detail::errormsg(
                "exception was raised in user function",
                names, x);
            throw std::runtime_error(msg);
        }

        const double ret = PyFloat_AsDouble(result);

        if (PyErr_Occurred()) {
            std::string msg = detail::errormsg(
                "cannot convert call result to double",
                names, x);
            throw std::runtime_error(msg);
        }

        if (detail::isnan(ret)) {
            std::string msg = detail::errormsg("result is NaN", names, x);
            if (throw_nan)
                throw std::runtime_error(msg);
        }

        ++ncall;
        return ret;
    }

    std::vector<double> vector(const std::vector<double>& x,
                               const std::vector<std::string>& names,
                               const bool throw_nan) const {
        PyHandle args, result;
        args = convert(x); // no error can occur here
        result = PyObject_CallObject(fcn, args);

        if (!result) {
            std::string msg = detail::errormsg(
                "exception was raised in user function",
                names, x);
            throw std::runtime_error(msg);
        }

        // Convert the iterable to a vector
        PyHandle iterator;
        iterator = PyObject_GetIter(result);
        if (!iterator) {
            std::string msg = detail::errormsg(
                "result must be iterable",
                names, x);
            throw std::runtime_error(msg);
        }

        std::vector<double> result_vector;
        result_vector.reserve(PySequence_Size(result));
        PyHandle item;
        while ((item = PyIter_Next(iterator))) {
            const double xi = PyFloat_AsDouble(item);

            if (PyErr_Occurred()) {
                std::string msg = detail::errormsg(
                    "cannot convert to vector of doubles",
                    names, x);
                throw std::runtime_error(msg);
            }

            if (detail::isnan(xi)) {
                std::string msg = detail::errormsg("result is NaN", names, x);
                if (throw_nan)
                    throw std::runtime_error(msg);
            }
            result_vector.push_back(xi);
        }

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
