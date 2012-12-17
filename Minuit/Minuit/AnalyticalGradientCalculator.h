#ifndef MN_AnalyticalGradientCalculator_H_
#define MN_AnalyticalGradientCalculator_H_

#include "Minuit/GradientCalculator.h"

class FCNGradientBase;
class MnUserTransformation;

class AnalyticalGradientCalculator : public GradientCalculator {

public:

  AnalyticalGradientCalculator(const FCNGradientBase& fcn, const MnUserTransformation& state) : theGradCalc(fcn), theTransformation(state) {}

  ~AnalyticalGradientCalculator() {}

  virtual FunctionGradient operator()(const MinimumParameters&) const;

  virtual FunctionGradient operator()(const MinimumParameters&,
				      const FunctionGradient&) const;

  virtual bool checkGradient() const;

private:

  const FCNGradientBase& theGradCalc;
  const MnUserTransformation& theTransformation;
};

#endif //MN_AnalyticalGradientCalculator_H_
