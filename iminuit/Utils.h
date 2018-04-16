#ifndef IMINUIT_UTILS_H
#define IMINUIT_UTILS_H

#include <string>
#include <cstdlib>
#include "Minuit2/MnApplication.h"
#include "Minuit2/FunctionMinimum.h"

using namespace ROOT::Minuit2;

//missing string printf
inline std::string format(const char* fmt, ...){
    std::string ret(512, '\0');
    va_list vl;
    va_start(vl, fmt);
    const int size = ret.size();
    const int nsize = std::vsnprintf(&ret[0], ret.size(), fmt, vl);
    ret.resize(nsize); // shrink the string to right size
    if (size <= nsize) { // resize string and try again
        std::vsnprintf(&ret[0], nsize, fmt, vl);
    }
    va_end(vl);
    return ret;
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
