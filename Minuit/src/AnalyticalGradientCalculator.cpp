#include "Minuit/AnalyticalGradientCalculator.h"
#include "Minuit/FCNGradientBase.h"
#include "Minuit/MnUserTransformation.h"
#include "Minuit/FunctionGradient.h"
#include "Minuit/MinimumParameters.h"
#include "Minuit/MnMatrix.h"

FunctionGradient AnalyticalGradientCalculator::operator()(const MinimumParameters& par) const {

  std::vector<double> grad = theGradCalc.gradient(theTransformation(par.vec()));
  assert(grad.size() == theTransformation.parameters().size());

  MnAlgebraicVector v(par.vec().size());
  for(unsigned int i = 0; i < par.vec().size(); i++) {
    unsigned int ext = theTransformation.extOfInt(i);
    if(theTransformation.parameter(ext).hasLimits()) {
      //double dd = (theTransformation.parameter(ext).upper() - theTransformation.parameter(ext).lower())*0.5*cos(par.vec()(i));
//       const ParameterTransformation * pt = theTransformation.transformation(ext); 
//       double dd = pt->dInt2ext(par.vec()(i), theTransformation.parameter(ext).lower(), theTransformation.parameter(ext).upper() );       
      double dd = theTransformation.dInt2Ext(i, par.vec()(i));
      v(i) = dd*grad[ext];
    } else {
      v(i) = grad[ext];
    }
  }

  return FunctionGradient(v);
}

FunctionGradient AnalyticalGradientCalculator::operator()(const MinimumParameters& par, const FunctionGradient&) const {

  return (*this)(par);
}

bool AnalyticalGradientCalculator::checkGradient() const {
  return theGradCalc.checkGradient();
}
