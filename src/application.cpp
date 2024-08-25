#include "pybind11.hpp"
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnApplication.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;

void bind_application(py::module m) {
  py::class_<MnApplication>(m, "MnApplication")

      .def("__call__", &MnApplication::operator())
      .def_property("precision", &MnApplication::Precision,
                    &MnApplication::SetPrecision)
      .def_property_readonly("strategy", &MnApplication::Strategy)

      ;
}
