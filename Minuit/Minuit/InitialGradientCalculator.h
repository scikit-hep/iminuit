#ifndef MN_InitialGradientCalculator_H_
#define MN_InitialGradientCalculator_H_

#include "Minuit/GradientCalculator.h"

class MnFcn;
class MnUserTransformation;
class MnMachinePrecision;
class MnStrategy;

class InitialGradientCalculator : public GradientCalculator {
  
public:
  
  InitialGradientCalculator(const MnFcn& fcn, const MnUserTransformation& par,
			    const MnStrategy& stra) : 
    theFcn(fcn), theTransformation(par), theStrategy(stra) {};
  
  virtual ~InitialGradientCalculator() {}

  virtual FunctionGradient operator()(const MinimumParameters&) const;

  virtual FunctionGradient operator()(const MinimumParameters&,
				      const FunctionGradient&) const;

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

#endif //MN_InitialGradientCalculator_H_
