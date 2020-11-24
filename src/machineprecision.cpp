#include <Minuit2/MnMachinePrecision.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;

void bind_machineprecision(py::module m) {
  py::class_<MnMachinePrecision>(m, "MnMachinePrecision")

      .def_property("eps", &MnMachinePrecision::Eps, &MnMachinePrecision::SetPrecision)
      .def_property_readonly("eps2", &MnMachinePrecision::Eps2)

      ;
}
