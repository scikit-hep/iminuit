#include <Minuit2/MnPrint.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;
using cstr = const char*;
using namespace pybind11::literals;

void bind_print(py::module m) {
  py::class_<MnPrint>(m, "MnPrint")

      .def(py::init<cstr, int>(), "prefix"_a, "level"_a)
      .def("error", &MnPrint::Error<cstr>)
      .def("warn", &MnPrint::Warn<cstr>)
      .def("info", &MnPrint::Info<cstr>)
      .def("debug", &MnPrint::Debug<cstr>)
      .def_property_static(
          "global_level", [](py::object) { return MnPrint::GlobalLevel(); },
          [](py::object, int x) { MnPrint::SetGlobalLevel(x); })

      ;
}
