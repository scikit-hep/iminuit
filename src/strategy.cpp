#include <Minuit2/MnStrategy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;

void set_strategy(MnStrategy& self, unsigned s) {
  switch (s) {
    case 0: self.SetLowStrategy(); break;
    case 1: self.SetMediumStrategy(); break;
    case 2: self.SetHighStrategy(); break;
    default: throw std::invalid_argument("invalid strategy");
  }
}

void bind_strategy(py::module m) {
  // TODO add more interface to tune strategy
  py::class_<MnStrategy>(m, "MnStrategy")

      .def(py::init<>())
      .def(py::init<unsigned>())
      .def_property("strategy", &MnStrategy::Strategy, set_strategy)

      ;

  py::implicitly_convertible<unsigned, MnStrategy>();
}
