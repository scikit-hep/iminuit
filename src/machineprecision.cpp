#include <Minuit2/MnMachinePrecision.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

namespace ROOT {
namespace Minuit2 {

bool operator==(const MnMachinePrecision& a, const MnMachinePrecision& b) {
  return a.Eps() == b.Eps() && a.Eps2() == b.Eps2();
}

} // namespace Minuit2
} // namespace ROOT

namespace py = pybind11;
using namespace ROOT::Minuit2;

void bind_machineprecision(py::module m) {
  py::class_<MnMachinePrecision>(m, "MnMachinePrecision")

      .def(py::init<>())

      .def_property("eps", &MnMachinePrecision::Eps, &MnMachinePrecision::SetPrecision)
      .def_property_readonly("eps2", &MnMachinePrecision::Eps2)

      .def(py::self == py::self)

      .def(py::pickle(
          [](const MnMachinePrecision& self) { return py::make_tuple(self.Eps()); },
          [](py::tuple tp) {
            MnMachinePrecision p;
            p.SetPrecision(tp[0].cast<double>());
            return p;
          }))

      ;
}
