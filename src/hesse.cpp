#include <Minuit2/MnHesse.h>
#include <Minuit2/FCNBase.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/ContoursError.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;

void bind_hesse(py::module m) {
  py::class_<MnHesse>(m, "MnHesse")

      .def(py::init<const MnStrategy&>())
      .def("__call__", py::overload<const FCNBase&, const MnUserParameterState&, unsigned>(&MnHesse::operator()))

      ;
}
