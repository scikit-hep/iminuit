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

bool equal(MnStrategy& self, unsigned s) { return self.Strategy() == s; }
bool not_equal(MnStrategy& self, unsigned s) { return self.Strategy() != s; }
bool less(MnStrategy& self, unsigned s) { return self.Strategy() < s; }
bool greater(MnStrategy& self, unsigned s) { return self.Strategy() > s; }
bool less_equal(MnStrategy& self, unsigned s) { return self.Strategy() <= s; }
bool greater_equal(MnStrategy& self, unsigned s) { return self.Strategy() >= s; }

void bind_strategy(py::module m) {
  // TODO add more interface to tune strategy
  py::class_<MnStrategy>(m, "MnStrategy")

      .def(py::init<>())
      .def(py::init<unsigned>())
      .def_property("strategy", &MnStrategy::Strategy, set_strategy)
      .def("__eq__", equal, py::is_operator())
      .def("__ne__", not_equal, py::is_operator())
      .def("__lt__", less, py::is_operator())
      .def("__gt__", greater, py::is_operator())
      .def("__le__", less_equal, py::is_operator())
      .def("__ge__", greater_equal, py::is_operator())

      ;

  py::implicitly_convertible<unsigned, MnStrategy>();
}
