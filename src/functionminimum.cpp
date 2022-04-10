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

void bind_functionminimum(py::module m) {
  py::class_<FunctionMinimum>(m, "FunctionMinimum")

      .def(py::init(&init))
      .def(py::init(&init2))

      .def_property_readonly("state", &FunctionMinimum::UserState)
      .def_property_readonly("edm", &FunctionMinimum::Edm)
      .def_property_readonly("fval", &FunctionMinimum::Fval)
      // TODO this needs to be fixed upstream in ROOT
      .def_property_readonly("is_valid",
                             [](const FunctionMinimum& self) {
                               return self.IsValid() && !std::isnan(self.Edm());
                             })
      // FunctionMinimum::HasValidParameters is not wrapped to Python
      .def_property_readonly("has_accurate_covar", &FunctionMinimum::HasAccurateCovar)
      .def_property_readonly("has_posdef_covar", &FunctionMinimum::HasPosDefCovar)
      .def_property_readonly("has_made_posdef_covar",
                             &FunctionMinimum::HasMadePosDefCovar)
      .def_property_readonly("hesse_failed", &FunctionMinimum::HesseFailed)
      .def_property_readonly("has_covariance", &FunctionMinimum::HasCovariance)
      // TODO this needs to be fixed upstream in ROOT
      .def_property_readonly("is_above_max_edm",
                             [](const FunctionMinimum& self) {
                               return self.IsAboveMaxEdm() || std::isnan(self.Edm());
                             })
      .def_property_readonly("has_reached_call_limit",
                             &FunctionMinimum::HasReachedCallLimit)
      .def_property("errordef", &FunctionMinimum::Up, &FunctionMinimum::SetErrorDef)

      .def(py::self == py::self)

      .def(py::pickle(&fmin_getstate, &fmin_setstate))

      ;
}
