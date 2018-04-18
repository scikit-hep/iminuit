#ifndef IMINUIT_UTILS_H
#define IMINUIT_UTILS_H

#include <string>
#include <stdio.h>
#include <stdarg.h>
#include <algorithm>
#include "Minuit2/MnApplication.h"
#include "Minuit2/FunctionMinimum.h"

using namespace ROOT::Minuit2;

//missing string printf
inline std::string format(const char* fmt, ...){
    const int size = strlen(fmt) * 2;
    char* buf = new char[strlen(fmt) * 2]; // reserve twice the length of fmt
    va_list vl;
    va_start(vl, fmt);
    const int nsize = vsnprintf(buf, size, fmt, vl);
    if (size <= nsize) { // resize string and try again
        delete [] buf;
        buf = new char[nsize];
        vsprintf(buf, fmt, vl);
    }
    va_end(vl);
    std::string s(buf, buf+nsize);
    delete [] buf;
    return s;
}

//mnapplication() returns stack allocated functionminimum but
//cython doesn't like it since it has no default constructor
//wrap this in a throw since it calls python function it will throw
//caller is responsible to clean up the object
inline FunctionMinimum* call_mnapplication_wrapper(MnApplication& app,unsigned int i, double tol){
    FunctionMinimum* ret = new FunctionMinimum(app(i,tol));
    return ret;
}



#endif
