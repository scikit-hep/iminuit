#include "Minuit/SimplexSeedGenerator.h"
#include "Minuit/MnUserParameterState.h"
#include "Minuit/MnFcn.h"
#include "Minuit/MinimumSeed.h"
#include "Minuit/MnStrategy.h"
#include "Minuit/InitialGradientCalculator.h"
#include "Minuit/VariableMetricEDMEstimator.h"

MinimumSeed SimplexSeedGenerator::operator()(const MnFcn& fcn, const GradientCalculator&, const MnUserParameterState& st, const MnStrategy& stra) const {

  unsigned int n = st.variableParameters();
  const MnMachinePrecision& prec = st.precision();

  // initial starting values
  MnAlgebraicVector x(n);
  for(unsigned int i = 0; i < n; i++) x(i) = st.intParameters()[i];
  double fcnmin = fcn(x);
  MinimumParameters pa(x, fcnmin);
  InitialGradientCalculator igc(fcn, st.trafo(), stra);
  FunctionGradient dgrad = igc(pa);
  MnAlgebraicSymMatrix mat(n);
  double dcovar = 1.;
  for(unsigned int i = 0; i < n; i++)	
    mat(i,i) = (fabs(dgrad.g2()(i)) > prec.eps2() ? 1./dgrad.g2()(i) : 1.);
  MinimumError err(mat, dcovar);
  double edm = VariableMetricEDMEstimator().estimate(dgrad, err);
  MinimumState state(pa, err, dgrad, edm, fcn.numOfCalls());
  
  return MinimumSeed(state, st.trafo());		     
}

MinimumSeed SimplexSeedGenerator::operator()(const MnFcn& fcn, const AnalyticalGradientCalculator& gc, const MnUserParameterState& st, const MnStrategy& stra) const {
  return (*this)(fcn, (const GradientCalculator&)(gc), st, stra);
}
