#include <Minuit2/MnUserCovariance.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include "equal.hpp"

namespace py = pybind11;
using namespace ROOT::Minuit2;

MnUserCovariance init(py::sequence seq, unsigned n) {
  return MnUserCovariance{py::cast<std::vector<double>>(seq), n};
}

void bind_usercovariance(py::module m) {
  py::class_<MnUserCovariance>(m, "MnUserCovariance")

      .def(py::init(&init))

      .def("__getitem__",
           [](const MnUserCovariance& self, py::object args) {
             const auto tup = py::cast<std::pair<int, int>>(args);
             return self(tup.first, tup.second);
           })

      .def_property_readonly("nrow", &MnUserCovariance::Nrow)

      .def(py::self == py::self)

      ;
}
