#include "equal.hpp"
#include <Minuit2/MinuitParameter.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

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

      .def(py::pickle(
          [](const MinuitParameter& self) {
            return py::make_tuple(self.Number(), self.GetName(), self.Value(),
                                  self.Error(), self.IsConst(), self.IsFixed(),
                                  self.LowerLimit(), self.UpperLimit(),
                                  self.HasLowerLimit(), self.HasUpperLimit());
          },
          [](py::tuple tp) {
            static_assert(std::is_standard_layout<MinuitParameter>(), "");

            if (tp.size() != 10) throw std::runtime_error("invalid state");

            MinuitParameter p{tp[0].cast<unsigned>(), tp[1].cast<std::string>(),
                              tp[2].cast<double>(), tp[3].cast<double>()};

            struct Layout {
              unsigned int fNum;
              double fValue;
              double fError;
              bool fConst;
              bool fFix;
              double fLoLimit;
              double fUpLimit;
              bool fLoLimValid;
              bool fUpLimValid;
              std::string fName;
            };

            auto d = reinterpret_cast<Layout*>(&p);

            d->fConst = tp[4].cast<bool>();
            d->fFix = tp[5].cast<bool>();
            d->fLoLimit = tp[6].cast<double>();
            d->fUpLimit = tp[7].cast<double>();
            d->fLoLimValid = tp[8].cast<bool>();
            d->fUpLimValid = tp[9].cast<bool>();

            return p;
          }))

      ;
}
