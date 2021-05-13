#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MinimumSeed.h>
#include <Minuit2/MinimumState.h>
#include <Minuit2/MnUserParameterState.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <type_traits>
#include <vector>

namespace py = pybind11;
using namespace ROOT::Minuit2;

py::tuple seed2py(const MinimumSeed& seed);

MinimumSeed py2seed(py::tuple tp);

py::tuple fmin_getstate(const FunctionMinimum& self) {
  return py::make_tuple(seed2py(self.Seed()), self.Up(), self.States(),
                        self.IsAboveMaxEdm(), self.HasReachedCallLimit(),
                        self.UserState());
}

FunctionMinimum fmin_setstate(py::tuple tp) {
  static_assert(std::is_standard_layout<FunctionMinimum>(), "");

  if (tp.size() != 6) throw std::runtime_error("invalid state");

  struct Layout {
    MinimumSeed fSeed;
    std::vector<MinimumState> fStates;
    double fErrorDef;
    bool fAboveMaxEdm;
    bool fReachedCallLimit;
    mutable MnUserParameterState fUserState;
  };

  auto seed = py2seed(tp[0]);
  auto up = tp[1].cast<double>();

  FunctionMinimum fm(seed, up);

  // evil workaround, will segfault or cause UB if source layout changes
  auto& ptr = reinterpret_cast<std::shared_ptr<Layout>&>(fm);
  auto d = ptr.get();

  d->fStates = tp[2].cast<std::vector<MinimumState>>();
  d->fAboveMaxEdm = tp[3].cast<bool>();
  d->fReachedCallLimit = tp[4].cast<bool>();
  d->fUserState = tp[5].cast<MnUserParameterState>();
  return fm;
}
