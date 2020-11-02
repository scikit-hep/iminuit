#include <Minuit2/MnApplication.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;
// using cstr = const char*;

void bind_application(py::module m) {
  py::class_<MnApplication>(m, "MnApplication")

      .def_property("precision", &MnApplication::Precision,
                    &MnApplication::SetPrecision)
      .def_property_readonly("strategy", &MnApplication::Strategy)

      ;
}
