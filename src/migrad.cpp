#include "fcn.hpp"
#include "pybind11.hpp"
#include <Minuit2/FCNBase.h>
#include <Minuit2/MnMigrad.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;

void bind_migrad(py::module m) {
  py::class_<MnMigrad, MnApplication>(m, "MnMigrad")

      .def(py::init<const FCN&, const MnUserParameterState&, const MnStrategy&>(),
           py::keep_alive<1, 2>())
      .def("set_print_level",
           [](MnMigrad& self, int lvl) {
             return self.Minimizer().Builder().SetPrintLevel(lvl);
           })

      ;
}
