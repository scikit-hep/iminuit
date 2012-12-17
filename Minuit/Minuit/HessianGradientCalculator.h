#ifndef MN_HessianGradientCalculator_H_
#define MN_HessianGradientCalculator_H_

#include "Minuit/GradientCalculator.h"
#include "Minuit/MnMatrix.h"
#include <utility>


class MnFcn;
class MnUserTransformation;
class MnMachinePrecision;
class MnStrategy;

/**
   HessianGradientCalculator: class to calculate gradient for Hessian
 */

class HessianGradientCalculator : public GradientCalculator {
  
public:
  
  HessianGradientCalculator(const MnFcn& fcn, const MnUserTransformation& par,
			    const MnStrategy& stra) : 
    theFcn(fcn), theTransformation(par), theStrategy(stra) {}
  
  virtual ~HessianGradientCalculator() {}

  virtual FunctionGradient operator()(const MinimumParameters&) const;

  virtual FunctionGradient operator()(const MinimumParameters&,
				      const FunctionGradient&) const;

  std::pair<FunctionGradient, MnAlgebraicVector> deltaGradient(const MinimumParameters&, const FunctionGradient&) const;

  const MnFcn& fcn() const {return theFcn;}
  const MnUserTransformation& trafo() const {return theTransformation;} 
  const MnMachinePrecision& precision() const;
  const MnStrategy& strategy() const {return theStrategy;}
 
  unsigned int ncycle() const;
  double stepTolerance() const;
  double gradTolerance() const;

private:

  const MnFcn& theFcn;
  const MnUserTransformation& theTransformation; 
  const MnStrategy& theStrategy;
};

#endif //MN_HessianGradientCalculator_H_
