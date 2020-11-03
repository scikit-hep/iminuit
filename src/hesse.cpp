#include <Minuit2/FCNBase.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnHesse.h>
#include <Minuit2/MnUserParameterState.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;

void bind_hesse(py::module m) {

  py::class_<MnHesse>(m, "MnHesse")

      .def(py::init<const MnStrategy&>())
      .def("__call__",
           [](MnHesse& self, const FCNBase& fcn, const MnUserParameterState& state,
              unsigned maxcalls) -> MnUserParameterState {
             return self(fcn, state, maxcalls);
           })
      .def("__call__", [](MnHesse& self, const FCNBase& fcn, FunctionMinimum& fm,
                          unsigned maxcalls) { self(fcn, fm, maxcalls); })

      ;
}
