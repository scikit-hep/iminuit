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
//wrap this in a throw since it calls python function it will throw
//caller is responsible to clean up the object
inline FunctionMinimum* call_mnapplication_wrapper(MnApplication& app,unsigned int i, double tol){
    FunctionMinimum* ret = new FunctionMinimum(app(i,tol));
    return ret;
}



#endif
