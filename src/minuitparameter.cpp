#include <Minuit2/MinuitParameter.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include "equal.hpp"

namespace ROOT {
namespace Minuit2 {

bool operator==(const MinuitParameter& a, const MinuitParameter& b) {
  return a.Number() == b.Number() && a.GetName() == b.GetName() &&
         a.Value() == b.Value() && a.Error() == b.Error() &&
         a.IsConst() == b.IsConst() && a.IsFixed() == b.IsFixed() &&
         a.HasLimits() == b.HasLimits() && a.HasLowerLimit() == b.HasLowerLimit() &&
         a.HasUpperLimit() == b.HasUpperLimit() &&
         nan_equal(a.LowerLimit(), b.LowerLimit()) &&
         nan_equal(a.UpperLimit(), b.UpperLimit());
}

} // namespace Minuit2
} // namespace ROOT

namespace py = pybind11;
using namespace ROOT::Minuit2;

void bind_minuitparameter(py::module m) {
  py::class_<MinuitParameter>(m, "MinuitParameter")

      .def_property_readonly("number", &MinuitParameter::Number)
      .def_property_readonly("name", &MinuitParameter::GetName)
      .def_property_readonly("value", &MinuitParameter::Value)
      .def_property_readonly("error", &MinuitParameter::Error)
      .def_property_readonly("is_const", &MinuitParameter::IsConst)
      .def_property_readonly("is_fixed", &MinuitParameter::IsFixed)
      .def_property_readonly("has_limits", &MinuitParameter::HasLimits)
      .def_property_readonly("has_lower_limit", &MinuitParameter::HasLowerLimit)
      .def_property_readonly("has_upper_limit", &MinuitParameter::HasUpperLimit)
      .def_property_readonly("lower_limit", &MinuitParameter::LowerLimit)
      .def_property_readonly("upper_limit", &MinuitParameter::UpperLimit)

      .def(py::self == py::self)

      ;
}
