#ifndef IMINUIT_UTILS_H
#define IMINUIT_UTILS_H

#include <string>
#include <stdio.h>
#include <stdarg.h>
#include <algorithm>
#include "Minuit2/MnApplication.h"
#include "Minuit2/FunctionMinimum.h"
#include <Python.h>

//missing string printf
inline std::string format(const char* fmt, ...){
    char buffer[256];
    va_list vl;
    va_start(vl, fmt);
    const int size = vsnprintf(buffer, 256, fmt, vl);
    if (256 <= size) { // resize string and try again
        char * buf = new char[size + 1];
        vsprintf(buf, fmt, vl);
        va_end(vl);
        std::string s(buf, buf + size);
        delete [] buf;
        return s;
    }
    va_end(vl);
    return std::string(buffer, buffer + size);
}

//mnapplication() returns stack allocated functionminimum but
//cython doesn't like it since it has no default constructor
inline ROOT::Minuit2::FunctionMinimum* call_mnapplication_wrapper(
        ROOT::Minuit2::MnApplication& app, unsigned int i, double tol){
    return new ROOT::Minuit2::FunctionMinimum(app(i, tol));
}

// Helper to enable scoped ref-counting, which greatly simplifies the code
// when handling exceptions; the concept is borrowed from Boost.Python
class PyHandle {
public:
    PyHandle() : ptr(NULL) {}
    PyHandle(const PyHandle& o) : ptr(o.ptr) {
        Py_INCREF(ptr);
    }
    PyHandle& operator=(const PyHandle& o) {
        if (this != &o) {
            this->~PyHandle();
            ptr = o.ptr;
            Py_INCREF(ptr);
        }
        return *this;
    }
    PyHandle(PyObject* o) : ptr(o) {}
    PyHandle& operator=(PyObject* o) {
        this->~PyHandle();
        ptr = o;
        return *this;
    }
    ~PyHandle() {
        Py_XDECREF(ptr);
    }
    operator PyObject* () { return ptr; }
    PyObject** operator&() { return &ptr; }
    operator bool() const { return bool(ptr); }
private:
    PyObject* ptr;
};

#endif
