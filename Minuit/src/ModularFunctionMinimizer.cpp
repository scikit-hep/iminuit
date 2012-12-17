#include "Minuit/ModularFunctionMinimizer.h"
#include "Minuit/MinimumSeedGenerator.h"
#include "Minuit/AnalyticalGradientCalculator.h"
#include "Minuit/Numerical2PGradientCalculator.h"
#include "Minuit/MinimumBuilder.h"
#include "Minuit/MinimumSeed.h"
#include "Minuit/FunctionMinimum.h"
#include "Minuit/MnUserParameterState.h"
#include "Minuit/MnUserParameters.h"
#include "Minuit/MnUserCovariance.h"
#include "Minuit/MnUserTransformation.h"
#include "Minuit/MnUserFcn.h"
#include "Minuit/FCNBase.h"
#include "Minuit/FCNGradientBase.h"
#include "Minuit/MnStrategy.h"
#include "Minuit/MnHesse.h"
#include "Minuit/MnLineSearch.h"
#include "Minuit/MnParabolaPoint.h"
#include "Minuit/FumiliFCNBase.h"
#include "Minuit/FumiliGradientCalculator.h"

// #include "Minuit/MnUserParametersPrint.h"

FunctionMinimum ModularFunctionMinimizer::minimize(const FCNBase& fcn, const std::vector<double>& par, const std::vector<double>& err, unsigned int stra, unsigned int maxfcn, double toler) const {

  MnUserParameterState st(par, err);
  MnStrategy strategy(stra);
  return minimize(fcn, st, strategy, maxfcn, toler);
}
  
FunctionMinimum ModularFunctionMinimizer::minimize(const FCNGradientBase& fcn, const std::vector<double>& par, const std::vector<double>& err, unsigned int stra, unsigned int maxfcn, double toler) const {
  MnUserParameterState st(par, err);
  MnStrategy strategy(stra);
  return minimize(fcn, st, strategy, maxfcn, toler);
}

// move nrow before cov to avoid ambiguities when using default parameters
FunctionMinimum ModularFunctionMinimizer::minimize(const FCNBase& fcn, const std::vector<double>& par, unsigned int nrow, const std::vector<double>& cov, unsigned int stra, unsigned int maxfcn, double toler) const {

  MnUserParameterState st(par, cov, nrow);
  MnStrategy strategy(stra);
  return minimize(fcn, st, strategy, maxfcn, toler);
}
 
FunctionMinimum ModularFunctionMinimizer::minimize(const FCNGradientBase& fcn, const std::vector<double>& par, unsigned int nrow, const std::vector<double>& cov, unsigned int stra, unsigned int maxfcn, double toler) const {

  MnUserParameterState st(par, cov, nrow);
  MnStrategy strategy(stra);
  return minimize(fcn, st, strategy, maxfcn, toler);
}

FunctionMinimum ModularFunctionMinimizer::minimize(const FCNBase& fcn, const MnUserParameters& upar, const MnStrategy& strategy, unsigned int maxfcn, double toler) const {

  MnUserParameterState st(upar);
  return minimize(fcn, st, strategy, maxfcn, toler);
}

FunctionMinimum ModularFunctionMinimizer::minimize(const FCNGradientBase& fcn, const MnUserParameters& upar, const MnStrategy& strategy, unsigned int maxfcn, double toler) const {

  MnUserParameterState st(upar);
  return minimize(fcn, st, strategy, maxfcn, toler);
}

FunctionMinimum ModularFunctionMinimizer::minimize(const FCNBase& fcn, const MnUserParameters& upar, const MnUserCovariance& cov, const MnStrategy& strategy, unsigned int maxfcn, double toler) const {

  MnUserParameterState st(upar, cov);
  return minimize(fcn, st, strategy, maxfcn, toler);
}

FunctionMinimum ModularFunctionMinimizer::minimize(const FCNGradientBase& fcn, const MnUserParameters& upar, const MnUserCovariance& cov, const MnStrategy& strategy, unsigned int maxfcn, double toler) const {

  MnUserParameterState st(upar, cov);
  return minimize(fcn, st, strategy, maxfcn, toler);
}



FunctionMinimum ModularFunctionMinimizer::minimize(const FCNBase& fcn, const MnUserParameterState& st, const MnStrategy& strategy, unsigned int maxfcn, double toler) const {

  // neeed MnUsserFcn for diference int-ext parameters
  MnUserFcn mfcn(fcn, st.trafo() );
  Numerical2PGradientCalculator gc(mfcn, st.trafo(), strategy);

  unsigned int npar = st.variableParameters();
  if(maxfcn == 0) maxfcn = 200 + 100*npar + 5*npar*npar;
  MinimumSeed mnseeds = seedGenerator()(mfcn, gc, st, strategy);

  return minimize(mfcn, gc, mnseeds, strategy, maxfcn, toler);
}


// use gradient here 
FunctionMinimum ModularFunctionMinimizer::minimize(const FCNGradientBase& fcn, const MnUserParameterState& st, const MnStrategy& strategy, unsigned int maxfcn, double toler) const {

  MnUserFcn mfcn(fcn, st.trafo());
  AnalyticalGradientCalculator gc(fcn, st.trafo());

  unsigned int npar = st.variableParameters();
  if(maxfcn == 0) maxfcn = 200 + 100*npar + 5*npar*npar;

  MinimumSeed mnseeds = seedGenerator()(mfcn, gc, st, strategy);

  return minimize(mfcn, gc, mnseeds, strategy, maxfcn, toler);
}

// function that actually do the work 

FunctionMinimum ModularFunctionMinimizer::minimize(const MnFcn& mfcn, const GradientCalculator& gc, const MinimumSeed& seed, const MnStrategy& strategy, unsigned int maxfcn, double toler) const {

  const MinimumBuilder & mb = builder();
  //std::cout << typeid(&mb).name() << std::endl;
  return mb.minimum(mfcn, gc, seed, strategy, maxfcn, toler*mfcn.up());
}



