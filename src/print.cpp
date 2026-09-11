#include "pybind11.hpp"
#include <Minuit2/MnPrint.h>
#include <string>

namespace py = pybind11;
using namespace ROOT::Minuit2;
using cstr = const char*;
using namespace pybind11::literals;

// MnPrint pushes the prefix pointer on a thread-local stack in its constructor
// and pops it in its destructor. Python destroys objects in arbitrary order and
// possibly in another thread, so we cannot hold an MnPrint member. Instead we
// store prefix and level and make a local MnPrint for each message, which
// keeps the push/pop properly nested.
struct PyMnPrint {
  std::string prefix;
  int level;

  PyMnPrint(std::string p, int l) : prefix(std::move(p)), level(l) {}
};

void bind_print(py::module m) {
  py::class_<PyMnPrint>(m, "MnPrint")

      .def(py::init<std::string, int>(), "prefix"_a, "level"_a)
      .def("error",
           [](PyMnPrint& self, cstr msg) {
             MnPrint print(self.prefix.c_str(), self.level);
             print.Error(msg);
           })
      .def("warn",
           [](PyMnPrint& self, cstr msg) {
             MnPrint print(self.prefix.c_str(), self.level);
             print.Warn(msg);
           })
      .def("info",
           [](PyMnPrint& self, cstr msg) {
             MnPrint print(self.prefix.c_str(), self.level);
             print.Info(msg);
           })
      .def("debug",
           [](PyMnPrint& self, cstr msg) {
             MnPrint print(self.prefix.c_str(), self.level);
             print.Debug(msg);
           })
      .def_property_static(
          "global_level", [](py::object) { return MnPrint::GlobalLevel(); },
          [](py::object, int x) { MnPrint::SetGlobalLevel(x); })

      .def_static("show_prefix_stack", &MnPrint::ShowPrefixStack)
      .def_static("add_filter", &MnPrint::AddFilter)
      .def_static("clear_filter", &MnPrint::ClearFilter)

      ;
}
