#include "pybind11.hpp"
#include <Minuit2/ContoursError.h>
#include <Minuit2/FCNBase.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnContours.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;

void bind_contours(py::module m) {
  py::class_<MnContours>(m, "MnContours")

      .def(py::init<const FCNBase&, const FunctionMinimum&, const MnStrategy&>(),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
      .def("__call__",
           [](const MnContours& self, unsigned ix, unsigned iy, unsigned npoints) {
             const auto ce = self.Contour(ix, iy, npoints);
             return py::make_tuple(ce.XMinosError(), ce.YMinosError(), ce());
           })

      ;
}
