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
  // We reset call counter here in contrast to MnHesse.cxx:83
  MnUserFcn mfcn(fcn, min.UserState().Trafo(), 0);

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
      .def("__call__", update_fmin)

      ;
}
