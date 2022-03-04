#include "equal.hpp"
#include "type_caster.hpp"
#include <Minuit2/MnUserCovariance.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace ROOT {
namespace Minuit2 {

bool operator==(const MnUserCovariance& a, const MnUserCovariance& b) {
  return a.Nrow() == b.Nrow() && a.Data() == b.Data();
}

} // namespace Minuit2
} // namespace ROOT

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

      .def(py::pickle(
          [](const MnUserCovariance& self) {
            return py::make_tuple(self.Data(), self.Nrow());
          },
          [](py::tuple tp) {
            return MnUserCovariance(tp[0].cast<std::vector<double>>(),
                                    tp[1].cast<double>());
          }))

      ;
}
