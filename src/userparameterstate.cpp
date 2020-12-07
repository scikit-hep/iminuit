#include <Minuit2/MnUserParameterState.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "equal.hpp"

namespace ROOT {
namespace Minuit2 {

bool operator==(const MnUserParameterState& a, const MnUserParameterState& b) {
  return a.MinuitParameters() == b.MinuitParameters() && a.Fval() == b.Fval() &&
         a.Covariance() == b.Covariance() &&
         a.GlobalCC().GlobalCC() == b.GlobalCC().GlobalCC() &&
         a.IntParameters() == b.IntParameters() &&
         a.IntCovariance().Data() == b.IntCovariance().Data() &&
         a.CovarianceStatus() == b.CovarianceStatus() && a.IsValid() == b.IsValid() &&
         a.HasCovariance() == b.HasCovariance() && a.HasGlobalCC() == b.HasGlobalCC() &&
         a.Fval() == b.Fval() && a.Edm() == b.Edm() && a.NFcn() == b.NFcn();
}

} // namespace Minuit2
} // namespace ROOT

namespace py = pybind11;
using namespace ROOT::Minuit2;

int size(const MnUserParameterState& self) {
  return static_cast<int>(self.MinuitParameters().size());
}

const MinuitParameter& getitem(const MnUserParameterState& self, int i) {
  const int n = size(self);
  if (i < 0) i += n;
  if (i >= n) throw py::index_error();
  return self.Parameter(i);
}

auto iter(const MnUserParameterState& self) {
  return py::make_iterator(self.MinuitParameters().begin(),
                           self.MinuitParameters().end());
}

py::object globalcc2py(const MnGlobalCorrelationCoeff& gcc) {
  if (gcc.IsValid()) return py::cast(gcc.GlobalCC());
  return py::cast(nullptr);
}

void bind_userparameterstate(py::module m) {
  py::class_<MnUserParameterState>(m, "MnUserParameterState")

      .def(py::init<>())
      .def(py::init<const MnUserParameterState&>())

      .def("add",
           py::overload_cast<const std::string&, double>(&MnUserParameterState::Add))
      .def("add", py::overload_cast<const std::string&, double, double>(
                      &MnUserParameterState::Add))
      .def("add", py::overload_cast<const std::string&, double, double, double, double>(
                      &MnUserParameterState::Add))
      .def("fix", py::overload_cast<unsigned>(&MnUserParameterState::Fix))
      .def("release", py::overload_cast<unsigned>(&MnUserParameterState::Release))
      .def("set_value",
           py::overload_cast<unsigned, double>(&MnUserParameterState::SetValue))
      .def("set_error",
           py::overload_cast<unsigned, double>(&MnUserParameterState::SetError))
      .def("set_limits", py::overload_cast<unsigned, double, double>(
                             &MnUserParameterState::SetLimits))
      .def("set_upper_limit",
           py::overload_cast<unsigned, double>(&MnUserParameterState::SetUpperLimit))
      .def("set_lower_limit",
           py::overload_cast<unsigned, double>(&MnUserParameterState::SetLowerLimit))
      .def("remove_limits",
           py::overload_cast<unsigned>(&MnUserParameterState::RemoveLimits))
      .def_property_readonly("fval", &MnUserParameterState::Fval)
      .def_property_readonly("edm", &MnUserParameterState::Edm)
      .def_property_readonly("covariance", &MnUserParameterState::Covariance)
      .def_property_readonly(
          "globalcc",
          [](const MnUserParameterState& self) { return globalcc2py(self.GlobalCC()); })
      .def_property_readonly("is_valid", &MnUserParameterState::IsValid)
      .def_property_readonly("has_covariance", &MnUserParameterState::HasCovariance)
      .def("__len__", size)
      .def("__getitem__", getitem)
      .def("__iter__", iter)
      .def(py::self == py::self)

      ;
}
