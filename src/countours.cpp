#include <Minuit2/ContoursError.h>
#include <Minuit2/FCNBase.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnContours.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;

void bind_contours(py::module m) {
  py::class_<MnContours>(m, "MnContours")

      .def(py::init<const FCNBase&, const FunctionMinimum&, const MnStrategy&>())
      .def("__call__",
           [](const MnContours& self, unsigned ix, unsigned iy, unsigned npoints) {
             const auto ce = self.Contour(ix, iy, npoints);
             return py::make_tuple(ce.XMinosError(), ce.YMinosError(), ce());
           })

      ;
}
