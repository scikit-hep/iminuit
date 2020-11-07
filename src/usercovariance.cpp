#include <Minuit2/MnUserCovariance.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;
using cstr = const char*;

void bind_usercovariance(py::module m) {
  py::class_<MnUserCovariance>(m, "MnUserCovariance")

      .def("__getitem__", [](const MnUserCovariance& self, py::object args) {
        const auto tup = py::cast<std::pair<int, int>>(args);
        return self(tup.first, tup.second);
      });
}
