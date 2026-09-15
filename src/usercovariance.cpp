#include "equal.hpp"
#include "pybind11.hpp"
#include <Minuit2/MnUserCovariance.h>
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

MnUserCovariance make_covariance(std::vector<double> data, unsigned n) {
  // Minuit2 does not check this and then reads past the end of the data
  if (data.size() != n * (n + 1) / 2)
    throw py::value_error("data length does not match n * (n + 1) / 2");
  return MnUserCovariance{std::move(data), n};
}

MnUserCovariance init(py::sequence seq, unsigned n) {
  return make_covariance(py::cast<std::vector<double>>(seq), n);
}

void bind_usercovariance(py::module m) {
  py::class_<MnUserCovariance>(m, "MnUserCovariance")

      .def(py::init(&init))

      .def("__getitem__",
           [](const MnUserCovariance& self, py::object args) {
             auto tup = py::cast<std::pair<int, int>>(args);
             const int n = static_cast<int>(self.Nrow());
             if (tup.first < 0) tup.first += n;
             if (tup.second < 0) tup.second += n;
             if (tup.first < 0 || tup.first >= n || tup.second < 0 || tup.second >= n)
               throw py::index_error();
             return self(tup.first, tup.second);
           })

      .def_property_readonly("nrow", &MnUserCovariance::Nrow)

      .def(py::self == py::self)

      .def(py::pickle(
          [](const MnUserCovariance& self) {
            return py::make_tuple(self.Data(), self.Nrow());
          },
          [](py::tuple tp) {
            if (tp.size() != 2)
              throw std::runtime_error("MnUserCovariance invalid state");
            return make_covariance(tp[0].cast<std::vector<double>>(),
                                   tp[1].cast<unsigned>());
          }))

      ;
}
