#ifndef MN_FumiliGradientCalculator_H_
#define MN_FumiliGradientCalculator_H_

#include "Minuit/GradientCalculator.h"
#include "Minuit/MnMatrix.h"


class FumiliFCNBase;
class MnUserTransformation;

class FumiliGradientCalculator : public GradientCalculator {

public:

  FumiliGradientCalculator(const FumiliFCNBase& fcn, const MnUserTransformation& state, int n) : 
    theFcn(fcn), 
    theTransformation(state), 
    theHessian(MnAlgebraicSymMatrix(n) ) 
  {}

  ~FumiliGradientCalculator() {}
	       
  FunctionGradient operator()(const MinimumParameters&) const;

  FunctionGradient operator()(const MinimumParameters&,
				      const FunctionGradient&) const;

  const MnUserTransformation& trafo() const {return theTransformation;} 

  const MnAlgebraicSymMatrix & hessian() const { return theHessian; }

private:

  const FumiliFCNBase& theFcn;
  const MnUserTransformation& theTransformation;
  mutable MnAlgebraicSymMatrix theHessian;

};

#endif //MN_FumiliGradientCalculator_H_
