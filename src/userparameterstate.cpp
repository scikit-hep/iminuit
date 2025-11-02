#include "equal.hpp"
#include "pybind11.hpp"
#include <Minuit2/MnUserParameterState.h>
#include <type_traits>

namespace ROOT {
namespace Minuit2 {

bool operator==(const MnUserParameterState& a, const MnUserParameterState& b) {
  return a.MinuitParameters() == b.MinuitParameters() && a.Fval() == b.Fval() &&
         a.Covariance() == b.Covariance() &&
         a.GlobalCC().GlobalCC() == b.GlobalCC().GlobalCC() &&
         a.IntParameters() == b.IntParameters() &&
         a.IntCovariance().Data() == b.IntCovariance().Data() &&
         a.CovarianceStatus() == b.CovarianceStatus() && a.IsValid() == b.IsValid() &&
         a.HasCovariance() == b.HasCovariance() && a.Fval() == b.Fval() &&
         a.Edm() == b.Edm() && a.NFcn() == b.NFcn();
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

MnGlobalCorrelationCoeff py2globalcc(py::object o) {
  static_assert(std::is_standard_layout<MnGlobalCorrelationCoeff>(), "");

  struct Layout {
    std::vector<double> fGlobalCC;
    bool fValid;
  };

  MnGlobalCorrelationCoeff c;
  auto d = reinterpret_cast<Layout*>(&c);
  if (!o.is_none()) {
    d->fGlobalCC = o.cast<std::vector<double>>();
    d->fValid = true;
  }
  return c;
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

      .def_property_readonly("trafo", &MnUserParameterState::Trafo)

      .def("__len__", size)
      .def("__getitem__", getitem)
      .def("__iter__", iter)
      .def(py::self == py::self)

      .def(py::pickle(
          [](const MnUserParameterState& self) {
            return py::make_tuple(self.IsValid(), self.HasCovariance(),
                                  self.CovarianceStatus(), self.Fval(), self.Edm(),
                                  self.NFcn(), self.Trafo(), self.Covariance(),
                                  self.IntParameters(), self.IntCovariance());
          },
          [](py::tuple tp) {
            static_assert(std::is_standard_layout<MnUserParameterState>(), "");
            static_assert(std::is_standard_layout<MnUserParameters>(), "");

            if (tp.size() != 10)
              throw std::runtime_error("MnUserParameterState invalid state");

            struct Layout {
              bool fValid;
              bool fCovarianceValid;
              int fCovStatus; // covariance matrix status
              double fFVal;
              double fEDM;
              unsigned int fNFcn;

              MnUserParameters fParameters;
              MnUserCovariance fCovariance;

              std::vector<double> fIntParameters;
              MnUserCovariance fIntCovariance;
            };

            MnUserParameterState st;

            // evil workaround, will segfault or cause UB if source layout changes
            auto d = reinterpret_cast<Layout*>(&st);

            d->fValid = tp[0].cast<bool>();
            d->fCovarianceValid = tp[1].cast<bool>();
            d->fCovStatus = tp[2].cast<int>();
            d->fFVal = tp[3].cast<double>();
            d->fEDM = tp[4].cast<double>();
            d->fNFcn = tp[5].cast<unsigned>();
            reinterpret_cast<MnUserTransformation&>(d->fParameters) =
                tp[6].cast<MnUserTransformation>();
            d->fCovariance = tp[7].cast<MnUserCovariance>();
            d->fIntParameters = tp[8].cast<std::vector<double>>();
            d->fIntCovariance = tp[9].cast<MnUserCovariance>();

            return st;
          }))

      ;
}
