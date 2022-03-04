#include "fcn.hpp"
#include <Minuit2/FCNBase.h>
#include <Minuit2/MnMigrad.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;

MnMigrad init(const FCN& fcn, const MnUserParameterState& state,
              const MnStrategy& str) {
  if (fcn.grad_.is_none()) {
    return MnMigrad(static_cast<const FCNBase&>(fcn), state, str);
  }
  return MnMigrad(static_cast<const FCNGradientBase&>(fcn), state, str);
}

void bind_migrad(py::module m) {
  py::class_<MnMigrad, MnApplication>(m, "MnMigrad")

      .def(py::init(&init), py::keep_alive<1, 2>())
      .def("set_print_level",
           [](MnMigrad& self, int lvl) {
             return self.Minimizer().Builder().SetPrintLevel(lvl);
           })

      ;
}
