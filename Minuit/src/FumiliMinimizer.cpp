#include "Minuit/MnConfig.h"
#include "Minuit/FumiliMinimizer.h"
#include "Minuit/MinimumSeedGenerator.h"
#include "Minuit/FumiliGradientCalculator.h"
#include "Minuit/Numerical2PGradientCalculator.h"
#include "Minuit/AnalyticalGradientCalculator.h"
#include "Minuit/MinimumBuilder.h"
#include "Minuit/MinimumSeed.h"
#include "Minuit/FunctionMinimum.h"
#include "Minuit/MnUserParameterState.h"
#include "Minuit/MnUserParameters.h"
#include "Minuit/MnUserTransformation.h"
#include "Minuit/MnUserFcn.h"
#include "Minuit/FumiliFCNBase.h"
#include "Minuit/FCNGradientBase.h"
#include "Minuit/MnStrategy.h"
#include "Minuit/MnPrint.h"

// for Fumili implement minimize here because need downcast 


FunctionMinimum FumiliMinimizer::minimize(const FCNBase& fcn, const MnUserParameterState& st, const MnStrategy& strategy, unsigned int maxfcn, double toler) const {

  MnUserFcn mfcn(fcn, st.trafo());
  Numerical2PGradientCalculator gc(mfcn, st.trafo(), strategy);

  unsigned int npar = st.variableParameters();
  if(maxfcn == 0) maxfcn = 200 + 100*npar + 5*npar*npar;
  
  MinimumSeed mnseeds = seedGenerator()(mfcn, gc, st, strategy);
  
  // downcast fcn 

  //std::cout << "FCN type " << typeid(&fcn).name() << std::endl;

  FumiliFCNBase * fumiliFcn = dynamic_cast< FumiliFCNBase *>( const_cast<FCNBase *>(&fcn) ); 
  if ( !fumiliFcn ) { 
    std::cout <<"FumiliMinimizer: Error : wrong FCN type. Try to use default minimizer" << std::endl;
    return  FunctionMinimum(mnseeds, fcn.up() );
  }
   

  FumiliGradientCalculator fgc(*fumiliFcn, st.trafo(), npar);
#ifdef DEBUG
  std::cout << "Minuit::minimize using FumiliMinimizer" << std::endl;
#endif 
  return ModularFunctionMinimizer::minimize(mfcn, fgc, mnseeds, strategy, maxfcn, toler);
}


// use gradient here 
FunctionMinimum FumiliMinimizer::minimize(const FCNGradientBase& fcn, const MnUserParameterState& st, const MnStrategy& strategy, unsigned int maxfcn, double toler) const {

  // need MnUserFcn
  MnUserFcn mfcn(fcn, st.trafo() );
  AnalyticalGradientCalculator gc(fcn, st.trafo());

  unsigned int npar = st.variableParameters();
  if(maxfcn == 0) maxfcn = 200 + 100*npar + 5*npar*npar;

  MinimumSeed mnseeds = seedGenerator()(mfcn, gc, st, strategy);

  // downcast fcn 

  FumiliFCNBase * fumiliFcn = dynamic_cast< FumiliFCNBase *>( const_cast<FCNGradientBase *>(&fcn) ); 
  if ( !fumiliFcn ) { 
    std::cout <<"FumiliMinimizer: Error : wrong FCN type. Try to use default minimizer" << std::endl;
    return  FunctionMinimum(mnseeds, fcn.up() );
  }
  

  FumiliGradientCalculator fgc(*fumiliFcn, st.trafo(), npar);
#ifdef DEBUG
  std::cout << "Minuit::minimize using FumiliMinimizer" << std::endl;
#endif
  return ModularFunctionMinimizer::minimize(mfcn, fgc, mnseeds, strategy, maxfcn, toler);

}
