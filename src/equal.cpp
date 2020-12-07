#include "equal.hpp"
#include <Minuit2/FunctionGradient.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MinimumError.h>
#include <Minuit2/MinimumParameters.h>
#include <Minuit2/MinimumSeed.h>
#include <Minuit2/MinuitParameter.h>
#include <Minuit2/MnMachinePrecision.h>
#include <Minuit2/MnMatrix.h>
#include <Minuit2/MnStrategy.h>
#include <Minuit2/MnUserParameterState.h>
#include <Minuit2/MnUserTransformation.h>
#include <algorithm>
#include <cmath>
#include <vector>

bool nan_equal(double a, double b) { return std::isnan(a) == std::isnan(b) || a == b; }

namespace std {
bool operator==(const vector<double>& a, const vector<double>& b) {
  return equal(a.begin(), a.end(), b.begin(), b.end());
}
} // namespace std

namespace ROOT {
namespace Minuit2 {

bool operator==(const MnStrategy& a, const MnStrategy& b) {
  return a.Strategy() == b.Strategy() && a.GradientNCycles() == b.GradientNCycles() &&
         a.GradientStepTolerance() == b.GradientStepTolerance() &&
         a.GradientTolerance() == b.GradientTolerance() &&
         a.HessianNCycles() == b.HessianNCycles() &&
         a.HessianStepTolerance() == b.HessianStepTolerance() &&
         a.HessianG2Tolerance() == b.HessianG2Tolerance() &&
         a.HessianGradientNCycles() == b.HessianGradientNCycles() &&
         a.StorageLevel() == b.StorageLevel();
}

bool operator==(const MinuitParameter& a, const MinuitParameter& b) {
  return a.Number() == b.Number() && a.GetName() == b.GetName() &&
         a.Value() == b.Value() && a.Error() == b.Error() &&
         a.IsConst() == b.IsConst() && a.IsFixed() == b.IsFixed() &&
         a.HasLimits() == b.HasLimits() && a.HasLowerLimit() == b.HasLowerLimit() &&
         a.HasUpperLimit() == b.HasUpperLimit() &&
         nan_equal(a.LowerLimit(), b.LowerLimit()) &&
         nan_equal(a.UpperLimit(), b.UpperLimit());
}

bool operator==(const MnUserCovariance& a, const MnUserCovariance& b) {
  return a.Nrow() == b.Nrow() && a.Data() == b.Data();
}

bool operator==(const MnUserParameterState& a, const MnUserParameterState& b) {
  return a.MinuitParameters() == b.MinuitParameters() && a.Fval() == b.Fval() &&
         a.Covariance() == b.Covariance() &&
         a.GlobalCC().GlobalCC() == b.GlobalCC().GlobalCC() &&
         a.IntParameters() == b.IntParameters() &&
         a.IntCovariance().Data() == b.IntCovariance().Data() &&
         a.CovarianceStatus() == b.CovarianceStatus() && a.IsValid() == b.IsValid() &&
         a.HasCovariance() == b.HasCovariance() && a.HasGlobalCC() == b.HasGlobalCC() &&
         a.Fval() == b.Fval() && a.Edm() == b.Edm() && a.NFcn() == b.NFcn();
}

bool operator==(const MnMachinePrecision& a, const MnMachinePrecision& b) {
  return a.Eps() == b.Eps() && a.Eps2() == b.Eps2();
}

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
