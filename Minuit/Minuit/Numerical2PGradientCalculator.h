#ifndef MN_Numerical2PGradientCalculator_H_
#define MN_Numerical2PGradientCalculator_H_

#include "Minuit/MnConfig.h"
#include "Minuit/GradientCalculator.h"
#include <vector>

class MnFcn;
class MnUserTransformation;
class MnMachinePrecision;
class MnStrategy;

class Numerical2PGradientCalculator : public GradientCalculator {
  
public:
  
  Numerical2PGradientCalculator(const MnFcn& fcn, 
				const MnUserTransformation& par,
				const MnStrategy& stra) : 
    theFcn(fcn), theTransformation(par), theStrategy(stra) {}
  
  virtual ~Numerical2PGradientCalculator() {}

  virtual FunctionGradient operator()(const MinimumParameters&) const;




  virtual FunctionGradient operator()(const std::vector<double>& params) const;




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

#endif //MN_Numerical2PGradientCalculator_H_
