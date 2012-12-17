#include "Minuit/MnSeedGenerator.h"
#include "Minuit/MinimumSeed.h"
#include "Minuit/MnFcn.h"
#include "Minuit/GradientCalculator.h"
#include "Minuit/InitialGradientCalculator.h"
#include "Minuit/MnUserTransformation.h"
#include "Minuit/MinimumParameters.h"
#include "Minuit/FunctionGradient.h"
#include "Minuit/MinimumError.h"
#include "Minuit/MnMatrix.h"
#include "Minuit/MnMachinePrecision.h"
#include "Minuit/MinuitParameter.h"
#include "Minuit/MnLineSearch.h"
#include "Minuit/MnParabolaPoint.h"
#include "Minuit/MinimumState.h"
#include "Minuit/MnUserParameterState.h"
#include "Minuit/MnStrategy.h"
#include "Minuit/MnHesse.h"
#include "Minuit/VariableMetricEDMEstimator.h"
#include "Minuit/NegativeG2LineSearch.h"
#include "Minuit/AnalyticalGradientCalculator.h"
#include "Minuit/Numerical2PGradientCalculator.h"
#include "Minuit/HessianGradientCalculator.h"
#include "Minuit/MnPrint.h"

#include <math.h>

MinimumSeed MnSeedGenerator::operator()(const MnFcn& fcn, const GradientCalculator& gc, const MnUserParameterState& st, const MnStrategy& stra) const {

  unsigned int n = st.variableParameters();
  const MnMachinePrecision& prec = st.precision();

  // initial starting values
  MnAlgebraicVector x(n);
  for(unsigned int i = 0; i < n; i++) x(i) = st.intParameters()[i];
  double fcnmin = fcn(x);
  MinimumParameters pa(x, fcnmin);
  FunctionGradient dgrad = gc(pa);
  MnAlgebraicSymMatrix mat(n);
  double dcovar = 1.;
  if(st.hasCovariance()) {
    for(unsigned int i = 0; i < n; i++)	
      for(unsigned int j = i; j < n; j++) mat(i,j) = st.intCovariance()(i,j);
    dcovar = 0.;
  } else {
    for(unsigned int i = 0; i < n; i++)	
      mat(i,i) = (fabs(dgrad.g2()(i)) > prec.eps2() ? 1./dgrad.g2()(i) : 1.);
  }
  MinimumError err(mat, dcovar);
  double edm = VariableMetricEDMEstimator().estimate(dgrad, err);
  MinimumState state(pa, err, dgrad, edm, fcn.numOfCalls());

  NegativeG2LineSearch ng2ls;
  if(ng2ls.hasNegativeG2(dgrad, prec)) {
    state = ng2ls(fcn, state, gc, prec);
  }

  if(stra.strategy() == 2 && !st.hasCovariance()) {
    //calculate full 2nd derivative
    MinimumState tmp = MnHesse(stra)(fcn, state, st.trafo());
    return MinimumSeed(tmp, st.trafo());
  }
  
  return MinimumSeed(state, st.trafo());
}


MinimumSeed MnSeedGenerator::operator()(const MnFcn& fcn, const AnalyticalGradientCalculator& gc, const MnUserParameterState& st, const MnStrategy& stra) const {

  unsigned int n = st.variableParameters();
  const MnMachinePrecision& prec = st.precision();

  // initial starting values
  MnAlgebraicVector x(n);
  for(unsigned int i = 0; i < n; i++) x(i) = st.intParameters()[i];
  double fcnmin = fcn(x);
  MinimumParameters pa(x, fcnmin);

  InitialGradientCalculator igc(fcn, st.trafo(), stra);
  FunctionGradient tmp = igc(pa);
  FunctionGradient grd = gc(pa);
  FunctionGradient dgrad(grd.grad(), tmp.g2(), tmp.gstep());
  
  if(gc.checkGradient()) {
    bool good = true;
    HessianGradientCalculator hgc(fcn, st.trafo(), MnStrategy(2));
    std::pair<FunctionGradient, MnAlgebraicVector> hgrd = hgc.deltaGradient(pa, dgrad);
    for(unsigned int i = 0; i < n; i++) {
      if(fabs(hgrd.first.grad()(i) - grd.grad()(i)) > hgrd.second(i)) {
	std::cout<<"gradient discrepancy of external parameter "<<st.trafo().extOfInt(i)<<" (internal parameter "<<i<<") too large."<<std::endl;
	good = false;
      }
    }
    if(!good) {
      std::cout<<"Minuit does not accept user specified gradient. To force acceptance, override 'virtual bool checkGradient() const' of FCNGradientBase.h in the derived class."<<std::endl;
      assert(good);
    }
  }
  
  MnAlgebraicSymMatrix mat(n);
  double dcovar = 1.;
  if(st.hasCovariance()) {
    for(unsigned int i = 0; i < n; i++)	
      for(unsigned int j = i; j < n; j++) mat(i,j) = st.intCovariance()(i,j);
    dcovar = 0.;
  } else {
    for(unsigned int i = 0; i < n; i++)	
      mat(i,i) = (fabs(dgrad.g2()(i)) > prec.eps2() ? 1./dgrad.g2()(i) : 1.);
  }
  MinimumError err(mat, dcovar);
  double edm = VariableMetricEDMEstimator().estimate(dgrad, err);
  MinimumState state(pa, err, dgrad, edm, fcn.numOfCalls());

  NegativeG2LineSearch ng2ls;
  if(ng2ls.hasNegativeG2(dgrad, prec)) {
    Numerical2PGradientCalculator ngc(fcn, st.trafo(), stra);
    state = ng2ls(fcn, state, ngc, prec);
  }

  if(stra.strategy() == 2 && !st.hasCovariance()) {
    //calculate full 2nd derivative
    MinimumState tmp = MnHesse(stra)(fcn, state, st.trafo());
    return MinimumSeed(tmp, st.trafo());
  }
  
  return MinimumSeed(state, st.trafo());
}
