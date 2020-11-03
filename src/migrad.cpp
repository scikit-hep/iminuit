#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnMigrad.h>
#include <pybind11/pybind11.h>
#include "fcn.hpp"

namespace py = pybind11;
using namespace ROOT::Minuit2;

MnMigrad init(const FCN& fcn, const MnUserParameterState& state,
              const MnStrategy& str) {
  if (fcn.grad_) {
    return MnMigrad(static_cast<const FCNGradientBase&>(fcn), state, str);
  }
  return MnMigrad(static_cast<const FCNBase&>(fcn), state, str);
}

void bind_migrad(py::module m) {
  py::class_<MnMigrad, MnApplication>(m, "MnMigrad")

      .def(py::init(&init))
      .def("__call__", &MnMigrad::operator())

      ;
}
