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
#include "equal.hpp"
#include "fcn.hpp"

namespace py = pybind11;
using namespace ROOT::Minuit2;

MinimumSeed make_seed(const FCN& fcn, const MnUserFcn& mfcn,
                      const MnUserParameterState& st, const MnStrategy& str);

FunctionMinimum init(const FCN& fcn, const MnUserParameterState& st,
                     const MnStrategy& str, double edm_goal);

py::tuple fmin_getstate(const FunctionMinimum&);
FunctionMinimum fmin_setstate(py::tuple);

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
      .def_property("errordef", &FunctionMinimum::Up, &FunctionMinimum::SetErrorDef)

      .def(py::self == py::self)

      .def(py::pickle(&fmin_getstate, &fmin_setstate))

      ;
}
