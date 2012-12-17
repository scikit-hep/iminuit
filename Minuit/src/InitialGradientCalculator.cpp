#include "Minuit/InitialGradientCalculator.h"
#include "Minuit/MnFcn.h"
#include "Minuit/MnUserTransformation.h"
#include "Minuit/MnMachinePrecision.h"
#include "Minuit/MinimumParameters.h"
#include "Minuit/FunctionGradient.h"
#include "Minuit/MnStrategy.h"

#include <math.h>

FunctionGradient InitialGradientCalculator::operator()(const MinimumParameters& par) const {

  assert(par.isValid());

  unsigned int n = trafo().variableParameters();
  assert(n == par.vec().size());

  MnAlgebraicVector gr(n), gr2(n), gst(n);

  // initial starting values
  for(unsigned int i = 0; i < n; i++) {
    unsigned int exOfIn = trafo().extOfInt(i);

    double var = par.vec()(i);
    double werr = trafo().parameter(exOfIn).error();
    double sav = trafo().int2ext(i, var); 
    double sav2 = sav + werr;
    if(trafo().parameter(exOfIn).hasLimits()) {
      if(trafo().parameter(exOfIn).hasUpperLimit() &&
	 sav2 > trafo().parameter(exOfIn).upperLimit()) 
 	sav2 = trafo().parameter(exOfIn).upperLimit();
    }
    double var2 = trafo().ext2int(exOfIn, sav2);
    double vplu = var2 - var;
    sav2 = sav - werr;
    if(trafo().parameter(exOfIn).hasLimits()) {
      if(trafo().parameter(exOfIn).hasLowerLimit() && 
	 sav2 < trafo().parameter(exOfIn).lowerLimit()) 
 	sav2 = trafo().parameter(exOfIn).lowerLimit();
    }
    var2 = trafo().ext2int(exOfIn, sav2);
    double vmin = var2 - var;
    double dirin = 0.5*(fabs(vplu) + fabs(vmin));
    double g2 = 2.0*theFcn.errorDef()/(dirin*dirin);
    double gsmin = 8.*precision().eps2()*(fabs(var) + precision().eps2());
    double gstep = std::max(gsmin, 0.1*dirin);
    double grd = g2*dirin;
    if(trafo().parameter(exOfIn).hasLimits()) {
      if(gstep > 0.5) gstep = 0.5;
    }
    gr(i) = grd;
    gr2(i) = g2;
    gst(i) = gstep;
  }

  return FunctionGradient(gr, gr2, gst);  
}

FunctionGradient InitialGradientCalculator::operator()(const MinimumParameters& par, const FunctionGradient&) const {

  return (*this)(par);
}

const MnMachinePrecision& InitialGradientCalculator::precision() const {
  return theTransformation.precision();
}

unsigned int InitialGradientCalculator::ncycle() const {
  return strategy().gradientNCycles();
}

double InitialGradientCalculator::stepTolerance() const {
  return strategy().gradientStepTolerance();
}

double InitialGradientCalculator::gradTolerance() const {
  return strategy().gradientTolerance();
}

