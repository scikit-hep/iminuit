#ifndef MN_MnCovarianceSqueeze_H_
#define MN_MnCovarianceSqueeze_H_

#include "Minuit/MnMatrix.h"

class MnUserCovariance;
class MinimumError;

class MnCovarianceSqueeze {

public:

  MnCovarianceSqueeze() {}

  ~MnCovarianceSqueeze() {}

  MnUserCovariance operator()(const MnUserCovariance&, unsigned int) const;

  MinimumError operator()(const MinimumError&, unsigned int) const;

  MnAlgebraicSymMatrix operator()(const MnAlgebraicSymMatrix&, unsigned int) const;

private:

};

#endif //MN_MnCovarianceSqueeze_H_
