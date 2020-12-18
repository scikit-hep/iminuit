#include <Minuit2/AnalyticalGradientCalculator.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MinimumState.h>
#include <Minuit2/MnSeedGenerator.h>
#include <Minuit2/MnStrategy.h>
#include <Minuit2/MnUserFcn.h>
#include <Minuit2/MnUserParameterState.h>
#include <Minuit2/Numerical2PGradientCalculator.h>
#include <pybind11/pybind11.h>
#include <type_traits>
#include <vector>
#include "equal.hpp"
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

FunctionMinimum init(const FCN& fcn, const MnUserParameterState& st,
                     const MnStrategy& str, double edm_goal) {
  MnUserFcn mfcn(fcn, st.Trafo());
  MinimumSeed seed = make_seed(fcn, mfcn, st, str);

  const auto& val = seed.Parameters().Vec();
  const auto n = seed.Trafo().VariableParameters();

  MnAlgebraicVector err(n);
  for (unsigned int i = 0; i < n; i++) {
    err(i) = std::sqrt(2. * mfcn.Up() * seed.Error().InvHessian()(i, i));
  }

  MinimumParameters minp(val, err, seed.Fval());
  std::vector<MinimumState> minstv(1, MinimumState(minp, seed.Edm(), fcn.nfcn_));
  if (minstv.back().Edm() < edm_goal) return FunctionMinimum(seed, minstv, fcn.Up());
  return FunctionMinimum(seed, minstv, fcn.Up(), FunctionMinimum::MnAboveMaxEdm());
}

py::tuple seed2py(const MinimumSeed& seed) {
  return py::make_tuple(seed.State(), seed.Trafo(), seed.IsValid());
}

MinimumSeed py2seed(py::tuple tp) {
  static_assert(std::is_standard_layout<MinimumSeed>(), "");
  static_assert(std::is_standard_layout<BasicMinimumSeed>(), "");

  MinimumSeed seed(tp[0].cast<MinimumState>(), tp[1].cast<MnUserTransformation>());

  struct Layout {
    MinimumState fState;
    MnUserTransformation fTrafo;
    bool fValid;
  };

  auto& ptr = reinterpret_cast<std::shared_ptr<BasicMinimumSeed>&>(seed);
  auto d = reinterpret_cast<Layout*>(ptr.get());
  d->fValid = tp[2].cast<bool>();

  return seed;
}

namespace ROOT {
namespace Minuit2 {

bool operator==(const MnUserTransformation& a, const MnUserTransformation& b) {
  return a.Precision() == b.Precision() && a.Parameters() == b.Parameters();
}

bool operator==(const LAVector& a, const LAVector& b) {
  return std::equal(a.Data(), a.Data() + a.size(), b.Data(), b.Data() + b.size());
}

bool operator==(const LASymMatrix& a, const LASymMatrix& b) {
  return std::equal(a.Data(), a.Data() + a.size(), b.Data(), b.Data() + b.size());
}

bool operator==(const MinimumError& a, const MinimumError& b) {
  return a.InvHessian() == b.InvHessian() && a.Dcovar() == b.Dcovar() &&
         a.IsValid() == b.IsValid() && a.IsPosDef() == b.IsPosDef() &&
         a.IsMadePosDef() == b.IsMadePosDef() && a.HesseFailed() == b.HesseFailed() &&
         a.InvertFailed() == b.InvertFailed() && a.IsAvailable() == b.IsAvailable();
}

bool operator==(const FunctionGradient& a, const FunctionGradient& b) {
  return a.Grad() == b.Grad() && a.G2() == b.G2() && a.Gstep() == b.Gstep() &&
         a.IsValid() == b.IsValid() && a.IsAnalytical() == b.IsAnalytical();
}

bool operator==(const MinimumParameters& a, const MinimumParameters& b) {
  return a.Vec() == b.Vec() && a.Dirin() == b.Dirin() && a.Fval() == b.Fval() &&
         a.IsValid() == b.IsValid() && a.HasStepSize() == b.HasStepSize();
}

bool operator==(const MinimumState& a, const MinimumState& b) {
  return a.Parameters() == b.Parameters() && a.Error() == b.Error() &&
         a.Gradient() == b.Gradient() && a.Fval() == b.Fval() && a.Edm() == b.Edm() &&
         a.NFcn() == b.NFcn();
}

bool operator==(const MinimumSeed& a, const MinimumSeed& b) {
  return a.State() == b.State() && a.Trafo() == b.Trafo() && a.IsValid() == b.IsValid();
}

bool operator==(const FunctionMinimum& a, const FunctionMinimum& b) {
  return a.Seed() == b.Seed() && a.Up() == b.Up() && a.States() == b.States() &&
         a.IsAboveMaxEdm() == b.IsAboveMaxEdm() &&
         a.HasReachedCallLimit() == b.HasReachedCallLimit() &&
         a.UserState() == b.UserState();
}

} // namespace Minuit2
} // namespace ROOT
