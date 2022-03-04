#include "lasymmatrix.hpp"
#include "lavector.hpp"
#include <Minuit2/MinimumState.h>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <type_traits>

namespace py = pybind11;
using namespace ROOT::Minuit2;

py::tuple par2py(const MinimumParameters& pars) {
  return py::make_tuple(lavector2py(pars.Vec()), lavector2py(pars.Dirin()), pars.Fval(),
                        pars.IsValid(), pars.HasStepSize());
}

MinimumParameters py2par(py::tuple tp) {
  static_assert(std::is_standard_layout<MinimumParameters>(), "");

  struct Layout {
    MnAlgebraicVector fParameters;
    MnAlgebraicVector fStepSize;
    double fFVal;
    bool fValid;
    bool fHasStep;
  };

  MinimumParameters pars(py2lavector(tp[0]), py2lavector(tp[1]), tp[2].cast<double>());

  // evil workaround, will segfault or cause UB if source layout changes
  auto& ptr = reinterpret_cast<std::shared_ptr<Layout>&>(pars);
  auto d = ptr.get();
  d->fValid = tp[3].cast<bool>();
  d->fHasStep = tp[4].cast<bool>();

  return pars;
}

py::tuple err2py(const MinimumError& err) {
  return py::make_tuple(lasymmatrix2py(err.InvHessian()), err.Dcovar(),
                        static_cast<int>(err.GetStatus()));
}

MinimumError py2err(py::tuple tp) {
  auto status = static_cast<MinimumError::Status>(tp[2].cast<int>());
  if (status == MinimumError::MnPosDef)
    return MinimumError(py2lasymmatrix(tp[0]), tp[1].cast<double>());
  return MinimumError(py2lasymmatrix(tp[0]), status);
}

py::tuple grad2py(const FunctionGradient& g) {
  return py::make_tuple(lavector2py(g.Grad()), lavector2py(g.G2()),
                        lavector2py(g.Gstep()), g.IsValid(), g.IsAnalytical());
}

FunctionGradient py2grad(py::tuple tp) {
  const auto& gr = py2lavector(tp[0]);
  const auto& g2 = py2lavector(tp[1]);
  const auto& st = py2lavector(tp[2]);
  const auto& valid = tp[3].cast<bool>();
  const auto& analytical = tp[4].cast<bool>();

  if (valid) {
    if (analytical)
      return FunctionGradient{gr};
    else
      return FunctionGradient{gr, g2, st};
  }
  return FunctionGradient{gr.size()};
}

void bind_minimumstate(py::module m) {
  py::class_<MinimumState>(m, "MinimumState")

      .def(py::init<unsigned>())

      .def_property_readonly(
          "vec", [](const MinimumState& self) { return lavector2py(self.Vec()); })
      .def_property_readonly("fval", &MinimumState::Fval)
      .def_property_readonly("edm", &MinimumState::Edm)
      .def_property_readonly("nfcn", &MinimumState::NFcn)
      .def_property_readonly("is_valid", &MinimumState::IsValid)
      .def_property_readonly("has_parameters", &MinimumState::HasParameters)
      .def_property_readonly("has_covariance", &MinimumState::HasCovariance)

      .def(py::pickle(
          [](const MinimumState& self) {
            return py::make_tuple(par2py(self.Parameters()), err2py(self.Error()),
                                  grad2py(self.Gradient()), self.Edm(), self.NFcn());
          },
          [](py::tuple tp) {
            static_assert(std::is_standard_layout<MinimumState>(), "");

            struct Layout {
              MinimumParameters fParameters;
              MinimumError fError;
              FunctionGradient fGradient;
              double fEDM;
              int fNFcn;
            };

            MinimumState st{0};

            // evil workaround, will segfault or cause UB if source layout changes
            auto& ptr = reinterpret_cast<std::shared_ptr<Layout>&>(st);
            auto d = ptr.get();
            d->fParameters = py2par(tp[0]);
            d->fError = py2err(tp[1]);
            d->fGradient = py2grad(tp[2]);
            d->fEDM = tp[3].cast<double>();
            d->fNFcn = tp[4].cast<int>();

            return st;
          }))

      ;
}
