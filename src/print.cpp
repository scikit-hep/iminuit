#include "pybind11.hpp"
#include <Minuit2/MnPrint.h>
#include <string>

namespace py = pybind11;
using namespace ROOT::Minuit2;
using cstr = const char*;
using namespace pybind11::literals;

// MnPrint stores the prefix as a `const char*` and keeps it on a stack for its
// whole lifetime. We therefore must keep the prefix string alive as long as the
// MnPrint object exists. This holder owns the string; since members are
// destroyed in reverse order of declaration, `print` is destroyed before
// `prefix`, so the pointer stays valid for MnPrint's lifetime.
struct PyMnPrint {
  std::string prefix;
  MnPrint print;

  PyMnPrint(std::string p, int level)
      : prefix(std::move(p)), print(prefix.c_str(), level) {}
};

void bind_print(py::module m) {
  py::class_<PyMnPrint>(m, "MnPrint")

      .def(py::init<std::string, int>(), "prefix"_a, "level"_a)
      .def("error", [](PyMnPrint& self, cstr msg) { self.print.Error(msg); })
      .def("warn", [](PyMnPrint& self, cstr msg) { self.print.Warn(msg); })
      .def("info", [](PyMnPrint& self, cstr msg) { self.print.Info(msg); })
      .def("debug", [](PyMnPrint& self, cstr msg) { self.print.Debug(msg); })
      .def_property_static(
          "global_level", [](py::object) { return MnPrint::GlobalLevel(); },
          [](py::object, int x) { MnPrint::SetGlobalLevel(x); })

      .def("show_prefix_stack", &MnPrint::ShowPrefixStack)
      .def("add_filter", &MnPrint::AddFilter)
      .def("clear_filter", &MnPrint::ClearFilter)

      ;
}
