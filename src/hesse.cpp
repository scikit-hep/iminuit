#include <Minuit2/FCNBase.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnHesse.h>
#include <Minuit2/MnUserFcn.h>
#include <Minuit2/MnUserParameterState.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace ROOT::Minuit2;

void update_fmin(MnHesse& self, const FCNBase& fcn, FunctionMinimum& min,
                 unsigned maxcalls, float maxedm) {
  MnUserFcn mfcn(fcn, min.UserState().Trafo(), min.NFcn());

  // Run Hesse
  MinimumState st = self(mfcn, min.State(), min.UserState().Trafo(), maxcalls);

  // Need to re-evalute status of minimum, EDM could now be over max EDM or
  // maxcalls could be exhausted, see MnMigrad.cxx:187
  const auto edm = st.Edm();
  if (edm > 10 * maxedm)
    min.Add(st, FunctionMinimum::MnAboveMaxEdm);
  else if (st.Error().HasReachedCallLimit())
    // communicate to user that call limit was reached in MnHesse
    min.Add(st, FunctionMinimum::MnReachedCallLimit);
  else if (st.Error().IsAvailable())
    min.Add(st);
}

void bind_hesse(py::module m) {

  py::class_<MnHesse>(m, "MnHesse")

      .def(py::init<const MnStrategy&>())
      // pybind11 needs help to figure out the return value that's why we use lambdas
      .def(
          "__call__",
          [](MnHesse& self, const FCNBase& fcn, const MnUserParameterState& state,
             unsigned maxcalls) -> MnUserParameterState {
            return self(fcn, state, maxcalls);
          },
          py::keep_alive<1, 2>())

      .def("__call__", update_fmin)

      ;
}
