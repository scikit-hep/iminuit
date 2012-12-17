#ifndef MN_MnUserFcn_H_
#define MN_MnUserFcn_H_

#include "Minuit/MnFcn.h"

class MnUserTransformation;

class MnUserFcn : public MnFcn {

public:

  MnUserFcn(const FCNBase& fcn, const MnUserTransformation& trafo) :
    MnFcn(fcn), theTransform(trafo) {}

  ~MnUserFcn() {}

  virtual double operator()(const MnAlgebraicVector&) const;

private:

  const MnUserTransformation& theTransform;
};

#endif //MN_MnUserFcn_H_
