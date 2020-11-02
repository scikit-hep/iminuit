#include <Minuit2/FunctionMinimum.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;
// using cstr = const char*;

void bind_functionminimum(py::module m) {
  py::class_<FunctionMinimum>(m, "FunctionMinimum")

      // .def_property_readonly("state", &MnApplication::State)
      // .def_property_readonly("parameters", &MnApplication::Parameters)
      // .def_property_readonly("covariance", &MnApplication::Covariance)
      // .def("add", py::overload_cast<cstr, double>(&MnApplication::Add))
      // .def("add", py::overload_cast<cstr, double, double>(&MnApplication::Add))
      // .def("add",
      //      py::overload_cast<cstr, double, double, double,
      //      double>(&MnApplication::Add))
      // .def("fix", &MnApplication::Fix)
      // .def("release", &MnApplication::Release)
      // .def("set_value", py::overload_cast<unsigned,
      // double>(&MnApplication::SetValue)) .def("set_error",
      // py::overload_cast<unsigned, double>(&MnApplication::SetError))
      // .def("set_error", py::overload_cast<unsigned,
      // double>(&MnApplication::SetError))
      ;
}

//
// cdef extern from "Minuit2/FunctionMinimum.h":
//     cdef cppclass FunctionMinimum:
//         FunctionMinimum(FunctionMinimum)
//         const MnUserParameterState& UserState()
//         const MnUserCovariance& UserCovariance()
//         # const_MinimumParameter parameters()
//         # const_MinimumError error()
//
//         double Fval()
//         double Edm()
//         int NFcn()
//
//         double Up()
//         bint HasValidParameters()
//         bint IsValid()
//         bint HasValidCovariance()
//         bint HasAccurateCovar()
//         bint HasPosDefCovar()
//         bint HasMadePosDefCovar()
//         bint HesseFailed()
//         bint HasCovariance()
//         bint HasReachedCallLimit()
//         bint IsAboveMaxEdm()
//
//         void SetErrorDef(double up)
