#include "pybind11.hpp"
#include <Minuit2/FCNBase.h>
#include <Minuit2/MnScan.h>
#include <Minuit2/MnUserParameterState.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;

void bind_scan(py::module m) {
  py::class_<MnScan, MnApplication>(m, "MnScan")

      .def(py::init<const FCNBase&, const MnUserParameterState&, const MnStrategy&>())

      ;
}
