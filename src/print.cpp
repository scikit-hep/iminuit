#include <Minuit2/MnPrint.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;

void bind_print(py::module m) {
  py::class_<MnPrint>(m, "MnPrint")

      .def_property_static("global_level", &MnPrint::GlobalLevel,
                           &MnPrint::SetGlobalLevel)

      ;
}
