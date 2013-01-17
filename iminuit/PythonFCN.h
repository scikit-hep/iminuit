#include <Python.h>
#include <vector>
#include <string>
#include "Minuit/FCNBase.h"
#include "Minuit/MnApplication.h"
#include <stdexcept>
using namespace std;

#include <string>
#include <cstdarg>
#include <algorithm>
#include <cstdio>
#include <cmath>
//missing string printf
//this is safe and convenient but not exactly efficient
inline std::string format(const char* fmt, ...){
    int size = 512;
    char* buffer = 0;
    buffer = new char[size];
    va_list vl;
    va_start(vl,fmt);
    int nsize = vsnprintf(buffer,size,fmt,vl);
    if(size<=nsize){//fail delete buffer and try again
        delete buffer; buffer = 0;
        buffer = new char[nsize+1];//+1 for /0
        nsize = vsnprintf(buffer,size,fmt,vl);
    }
    std::string ret(buffer);
    va_end(vl);
    delete buffer;
    return ret;
}

int raise_py_err(){
    try{
        if(PyErr_Occurred()){
            return NULL;
        }else{
            throw;
        }
    }catch(const std::exception& exn){
        PyErr_SetString(PyExc_RuntimeError, exn.what());
        return NULL;
    }
    return NULL;
}

//mnapplication() returns stack allocated functionminimum but
//cython doesn't like it since it has no default constructor
//wrap this in a throw since it calls python function it will throw
//caller is responsible to clean up the object
FunctionMinimum* call_mnapplication_wrapper(MnApplication& app,unsigned int i, double tol){
    FunctionMinimum* ret = new FunctionMinimum(app(i,tol));
    return ret;
}

class PythonFCN:public FCNBase{
public:
    PyObject* fcn;
    double up_parm;
    vector<string> pname;
    bool thrownan;
    mutable unsigned int ncall; 
    
    PythonFCN():fcn(), up_parm(), pname(), 
    thrownan(), ncall(0)
    {}//for cython stack allocate but don't call this

    PythonFCN(PyObject* fcn,
        double up_parm,
        const vector<string>& pname,
        bool thrownan = false)
        :fcn(fcn),up_parm(up_parm),pname(pname),
        thrownan(thrownan), ncall(0)
    {
        Py_INCREF(fcn);
    }

    PythonFCN(const PythonFCN& pfcn)
        :fcn(pfcn.fcn),up_parm(pfcn.up_parm),pname(pfcn.pname),
        thrownan(pfcn.thrownan), ncall(pfcn.ncall)
    {
        Py_INCREF(fcn);
    }

    virtual ~PythonFCN()
    {
        Py_DECREF(fcn);
    }

    virtual double operator()(const std::vector<double>& x) const{
        //pack in tuple
        PyObject* tuple = vector2tuple(x);
        //call
        PyObject* result = PyObject_Call(fcn,tuple,NULL);
        //check result exception etc
        PyObject* exc = NULL;
        if((exc = PyErr_Occurred())){
            string msg = "Exception Occured \n"+errormsg(x);
            warn_preserve_error(msg.c_str());
            throw runtime_error(msg);
        }

        double ret = PyFloat_AsDouble(result);
        if((exc = PyErr_Occurred())){
            string msg = "Cannot convert fcn(*arg) to double \n"+errormsg(x);
            warn_preserve_error(msg.c_str());
            throw runtime_error(msg);
        }

        if(ret!=ret){//check if nan
            string msg = "fcn returns Nan\n"+errormsg(x);
            warn_preserve_error(msg.c_str());
            if(thrownan){
                PyErr_SetString(PyExc_RuntimeError,msg.c_str());
                throw runtime_error(msg.c_str());
            }
        }

        Py_DECREF(tuple);
        Py_DECREF(result);
        ncall++;
        return ret;
    }

    //warn but do not reset the error flag
    inline void warn_preserve_error(const string& msg)const{
        PyObject *ptype,*pvalue,*ptraceback;
        PyErr_Fetch(&ptype,&pvalue,&ptraceback);
        PyErr_Warn(NULL,msg.c_str());
        PyErr_Restore(ptype,pvalue,ptraceback);
    }

    inline string errormsg(const std::vector<double>& x) const{
        string ret = "fcn is called with following arguments:\n";
        assert(pname.size()==x.size());
        //determine longest variable length
        size_t maxlength = 0;
        for(int i=0;i<x.size();i++){
            maxlength = max(pname[i].size(),maxlength);
        }
        for(int i=0;i<x.size();i++){
            string line = format("%*s = %+f\n",maxlength+4,pname[i].c_str(),x[i]);
            ret += line;
        }
        return ret;
    }

    inline PyObject* vector2tuple(const std::vector<double>& x) const{
        //new reference
        PyObject* tuple = PyTuple_New(x.size());
        for(int i=0;i<x.size();++i){
            PyObject* data = PyFloat_FromDouble(x[i]);
            PyTuple_SET_ITEM(tuple,i,data); //steal ref
        }
        return tuple;
    }
    int getNumCall() const{return ncall;}
    void resetNumCall(){ncall = 0;}
    void set_up(double up){up_parm = up;}
    virtual double up() const{return up_parm;}
};
