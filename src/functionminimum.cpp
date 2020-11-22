#include <Minuit2/AnalyticalGradientCalculator.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnStrategy.h>
#include <Minuit2/MnUserFcn.h>
#include <Minuit2/MnUserParameterState.h>
#include <Minuit2/Numerical2PGradientCalculator.h>
#include <Minuit2/SimplexSeedGenerator.h>
#include <pybind11/pybind11.h>
#include "fcn.hpp"

namespace py = pybind11;
using namespace ROOT::Minuit2;

FunctionMinimum init(const FCN& fcn, const MnUserParameterState& st, double fval,
                     double edm, int nfcn) {
  SimplexSeedGenerator gen;
  MnStrategy str;
  MnUserFcn mfcn(fcn, st.Trafo());
  std::vector<MinimumState> minstv(1, MinimumState(fval, edm, nfcn));
  if (fcn.grad_.is_none()) {
    Numerical2PGradientCalculator gc(mfcn, st.Trafo(), str);
    MinimumSeed seed = gen(mfcn, gc, st, str);
    return FunctionMinimum(seed, minstv, fcn.Up());
  }
  AnalyticalGradientCalculator gc(fcn, st.Trafo());
  MinimumSeed seed = gen(mfcn, gc, st, str);
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
