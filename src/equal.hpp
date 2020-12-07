#ifndef IMINUIT_EQUAL_HPP
#define IMINUIT_EQUAL_HPP

#include <vector>

bool nan_equal(double a, double b);

namespace std {
bool operator==(const vector<double>& a, const vector<double>& b);
} // namespace std

namespace ROOT {
namespace Minuit2 {

class MnStrategy;
bool operator==(const MnStrategy& a, const MnStrategy& b);

class MinuitParameter;
bool operator==(const MinuitParameter& a, const MinuitParameter& b);

class MnUserCovariance;
bool operator==(const MnUserCovariance& a, const MnUserCovariance& b);

class MnUserParameterState;
bool operator==(const MnUserParameterState& a, const MnUserParameterState& b);

class MnMachinePrecision;
bool operator==(const MnMachinePrecision& a, const MnMachinePrecision& b);

class MnUserTransformation;
bool operator==(const MnUserTransformation& a, const MnUserTransformation& b);

class LAVector;
bool operator==(const LAVector& a, const LAVector& b);

class LASymMatrix;
bool operator==(const LASymMatrix& a, const LASymMatrix& b);

class MinimumError;
bool operator==(const MinimumError& a, const MinimumError& b);

class FunctionGradient;
bool operator==(const FunctionGradient& a, const FunctionGradient& b);

class MinimumParameters;
bool operator==(const MinimumParameters& a, const MinimumParameters& b);

class MinimumState;
bool operator==(const MinimumState& a, const MinimumState& b);

class MinimumSeed;
bool operator==(const MinimumSeed& a, const MinimumSeed& b);

class FunctionMinimum;
bool operator==(const FunctionMinimum& a, const FunctionMinimum& b);

} // namespace Minuit2
} // namespace ROOT

#endif
