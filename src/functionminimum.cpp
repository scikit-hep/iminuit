#include <Minuit2/AnalyticalGradientCalculator.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnSeedGenerator.h>
#include <Minuit2/MnStrategy.h>
#include <Minuit2/MnUserFcn.h>
#include <Minuit2/MnUserParameterState.h>
#include <Minuit2/Numerical2PGradientCalculator.h>
#include <pybind11/pybind11.h>
#include "fcn.hpp"

namespace py = pybind11;
using namespace ROOT::Minuit2;

MinimumSeed make_seed(const FCN& fcn, const MnUserFcn& mfcn,
                      const MnUserParameterState& st, const MnStrategy& str) {
  MnSeedGenerator gen;
  if (fcn.grad_.is_none()) {
    Numerical2PGradientCalculator gc(mfcn, st.Trafo(), str);
    return gen(mfcn, gc, st, str);
  }
  AnalyticalGradientCalculator gc(fcn, st.Trafo());
  return gen(mfcn, gc, st, str);
}

FunctionMinimum init(const FCN& fcn, const MnUserParameterState& st, double fval,
                     const MnStrategy& str) {
  MnUserFcn mfcn(fcn, st.Trafo());
  MinimumSeed seed = make_seed(fcn, mfcn, st, str);

  const auto& val = seed.Parameters().Vec();
  const auto n = seed.Trafo().VariableParameters();

  MnAlgebraicVector err(n);
  for (unsigned int i = 0; i < n; i++) {
    err(i) = std::sqrt(2. * mfcn.Up() * seed.Error().InvHessian()(i, i));
  }

  MinimumParameters minp(val, err, fval);
  std::vector<MinimumState> minstv(1, MinimumState(minp, seed.Edm(), fcn.nfcn_));
  return FunctionMinimum(seed, minstv, fcn.Up());
}

void bind_functionminimum(py::module m) {
  py::class_<FunctionMinimum>(m, "FunctionMinimum")

      .def(py::init(&init))

      .def_property_readonly("state", &FunctionMinimum::UserState)
      .def_property_readonly("edm", &FunctionMinimum::Edm)
      .def_property_readonly("fval", &FunctionMinimum::Fval)
      .def_property_readonly("is_valid", &FunctionMinimum::IsValid)
      .def_property_readonly("has_valid_parameters",
                             &FunctionMinimum::HasValidParameters)
      .def_property_readonly("has_accurate_covar", &FunctionMinimum::HasAccurateCovar)
      .def_property_readonly("has_posdef_covar", &FunctionMinimum::HasPosDefCovar)
      .def_property_readonly("has_made_posdef_covar",
                             &FunctionMinimum::HasMadePosDefCovar)
      .def_property_readonly("hesse_failed", &FunctionMinimum::HesseFailed)
      .def_property_readonly("has_covariance", &FunctionMinimum::HasCovariance)
      .def_property_readonly("is_above_max_edm", &FunctionMinimum::IsAboveMaxEdm)
      .def_property_readonly("has_reached_call_limit",
                             &FunctionMinimum::HasReachedCallLimit)
      .def_property("up", &FunctionMinimum::Up, &FunctionMinimum::SetErrorDef)

      ;
}
