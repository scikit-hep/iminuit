#include <vector>
#include <string>
#include <stdexcept>
using namespace std;
#include <string>
#include <cstdarg>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <Python.h>
#include "Minuit2/FCNGradientBase.h"
#include "CallCounterMixin.h"

using namespace ROOT::Minuit2;

class PythonGradientFCN: public FCNGradientBase, public CallCounterMixin {
public:
    PyObject* fcn;
    PyObject* gradfcn;
    double up_parm;
    vector<string> pname;
    bool thrownan;

    PythonGradientFCN()
        : fcn(), gradfcn(), up_parm(), pname(), thrownan()
    {}//for cython stack allocate but don't call this

    PythonGradientFCN(PyObject* fcn,
        PyObject* gradfcn,
        double up_parm,
        const vector<string>& pname,
        bool thrownan = false)
        : fcn(fcn), gradfcn(gradfcn), up_parm(up_parm), pname(pname),
          thrownan(thrownan)
    {
        Py_INCREF(fcn);
        Py_INCREF(gradfcn);
    }

    PythonGradientFCN(const PythonGradientFCN& pfcn)
        : CallCounterMixin(pfcn.getNumCall()),
          fcn(pfcn.fcn), gradfcn(pfcn.gradfcn), up_parm(pfcn.up_parm),
          pname(pfcn.pname), thrownan(pfcn.thrownan)
    {
        Py_INCREF(fcn);
        Py_INCREF(gradfcn);
    }

    virtual ~PythonGradientFCN()
    {
        Py_DECREF(fcn);
        Py_DECREF(gradfcn);
    }

    virtual double operator()(const std::vector<double>& x) const{
        //pack in tuple
        PyObject* tuple = vector2tuple(x);
        //call
        PyObject* result = PyObject_Call(fcn,tuple,NULL);
        //check result exception etc
        PyObject* exc = NULL;
        if((exc = PyErr_Occurred())){
            string msg = "Exception Occured \n"+errormsg("fcn", x);
            warn_preserve_error(msg.c_str());
            throw runtime_error(msg);
        }

        double ret = PyFloat_AsDouble(result);
        if((exc = PyErr_Occurred())){
            string msg = "Cannot convert fcn(*arg) to double \n"+errormsg("fcn", x);
            warn_preserve_error(msg.c_str());
            throw runtime_error(msg);
        }

        if(ret!=ret){//check if nan
            string msg = "fcn returns Nan\n"+errormsg("fcn", x);
            warn_preserve_error(msg.c_str());
            if(thrownan){
                PyErr_SetString(PyExc_RuntimeError,msg.c_str());
                throw runtime_error(msg.c_str());
            }
        }

        Py_DECREF(tuple);
        Py_DECREF(result);
        increaseNumCall();
        return ret;
    }

    virtual std::vector<double> Gradient(const std::vector<double>& x) const{
        //pack in tuple
        PyObject* tuple = vector2tuple(x);
        //call
        PyObject* result = PyObject_Call(gradfcn,tuple,NULL);
        //check result exception etc
        PyObject* exc = NULL;
        if((exc = PyErr_Occurred())){
            string msg = "Exception Occured \n"+errormsg("gradfcn", x);
            warn_preserve_error(msg.c_str());
            throw runtime_error(msg);
        }
        // Convert the iterable to a vector
        PyObject *iterator = PyObject_GetIter(result);
        PyObject *item;

        if (iterator == NULL) {
            string msg = "The result of gradfcn(*arg) must be iterable \n"+errormsg("gradfcn", x);
            warn_preserve_error(msg.c_str());
            throw runtime_error(msg);
        }

        std::vector<double> result_vector;
        while (item = PyIter_Next(iterator)) {
            result_vector.push_back(PyFloat_AsDouble(item));
            Py_DECREF(item);
        }

        Py_DECREF(iterator);

        if((exc = PyErr_Occurred())){
            string msg = "Cannot convert gradfcn(*arg) to a vector of doubles \n"+errormsg("gradfcn", x);
            warn_preserve_error(msg.c_str());
            throw runtime_error(msg);
        }

        Py_DECREF(tuple);
        Py_DECREF(result);
        return result_vector;
    }

    //warn but do not reset the error flag
    inline void warn_preserve_error(const string& msg)const{
        PyObject *ptype,*pvalue,*ptraceback;
        PyErr_Fetch(&ptype,&pvalue,&ptraceback);
        PyErr_Warn(NULL,msg.c_str());
        PyErr_Restore(ptype,pvalue,ptraceback);
    }

    inline string errormsg(const char* which, const std::vector<double>& x) const{
        string ret = which;
        ret += " is called with following arguments:\n";
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

    virtual double Up() const { return up_parm; }
    virtual void SetErrorDef(double up) { up_parm = up; }
};
