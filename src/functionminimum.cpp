#include "equal.hpp"
#include "fcn.hpp"
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MinimumSeed.h>
#include <Minuit2/MinimumState.h>
#include <Minuit2/MnStrategy.h>
#include <Minuit2/MnUserFcn.h>
#include <Minuit2/MnUserParameterState.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <type_traits>
#include <vector>

namespace py = pybind11;
using namespace ROOT::Minuit2;

FunctionMinimum init(const FCN& fcn, const MnUserParameterState& st,
                     const MnStrategy& str, double edm_goal);

FunctionMinimum init2(const MnUserTransformation& tr, py::sequence par,
                      py::sequence cov, py::sequence grad, double fval, double up,
                      double edm_goal, int nfcn, int max_nfcn, bool exact_hess_inv);

py::tuple fmin_getstate(const FunctionMinimum&);
FunctionMinimum fmin_setstate(py::tuple);

bool hesse_failed_workaround(const FunctionMinimum& self) {
  // Calling FunctionMinimum::HesseFailed is not reliable.
  // It calls MinimumError::HesseFailed, but this is false if
  // there were the failures MinimumError::MnReachedCallLimit
  // or MinimumError::MnInvertFailed, which cannot be queried
  // from FunctionMinimum.
  //
  // As a workaround, we return true if covariance exists, but
  // is not accurate or made positive definite.
  return self.HasCovariance() && !(self.HasPosDefCovar() || self.HasMadePosDefCovar());
}

void bind_functionminimum(py::module m) {
  py::class_<FunctionMinimum>(m, "FunctionMinimum")

      .def(py::init(&init))
      .def(py::init(&init2))

      .def_property_readonly("state", &FunctionMinimum::UserState)
      .def_property_readonly("edm", &FunctionMinimum::Edm)
      .def_property_readonly("fval", &FunctionMinimum::Fval)
      .def_property_readonly("is_valid", &FunctionMinimum::IsValid)
      // No need to wrap HasValidParameters
      // .def_property_readonly("has_valid_parameters",
      //                        &FunctionMinimum::HasValidParameters)
      .def_property_readonly("has_accurate_covar", &FunctionMinimum::HasAccurateCovar)
      .def_property_readonly("has_posdef_covar", &FunctionMinimum::HasPosDefCovar)
      .def_property_readonly("has_made_posdef_covar",
                             &FunctionMinimum::HasMadePosDefCovar)
      .def_property_readonly("hesse_failed", hesse_failed_workaround)
      .def_property_readonly("has_covariance", &FunctionMinimum::HasCovariance)
      .def_property_readonly("is_above_max_edm", &FunctionMinimum::IsAboveMaxEdm)
      .def_property_readonly("has_reached_call_limit",
                             &FunctionMinimum::HasReachedCallLimit)
      .def_property("errordef", &FunctionMinimum::Up, &FunctionMinimum::SetErrorDef)

      .def(py::self == py::self)

      .def(py::pickle(&fmin_getstate, &fmin_setstate))

      ;
}
