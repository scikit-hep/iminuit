#include <Minuit2/MinuitParameter.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;
using cstr = const char*;

void bind_minuitparameter(py::module m) {
  py::class_<MinuitParameter>(m, "MinuitParameter")

      .def(py::init<>())
      .def(py::init<unsigned, cstr, double>())
      .def_property_readonly("number", &MinuitParameter::Number)
      .def_property_readonly("name", &MinuitParameter::Name)
      .def_property_readonly("value", &MinuitParameter::Value)
      .def_property_readonly("error", &MinuitParameter::Error)
      .def_property_readonly("is_const", &MinuitParameter::IsConst)
      .def_property_readonly("is_fixed", &MinuitParameter::IsFixed)
      .def_property_readonly("has_limits", &MinuitParameter::HasLimits)
      .def_property_readonly("has_lower_limit", &MinuitParameter::HasLowerLimit)
      .def_property_readonly("has_upper_limit", &MinuitParameter::HasUpperLimit)
      .def_property_readonly("lower_limit", &MinuitParameter::LowerLimit)
      .def_property_readonly("upper_limit", &MinuitParameter::UpperLimit)

      ;
}
