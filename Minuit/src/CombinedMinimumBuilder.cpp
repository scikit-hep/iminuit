#include "Minuit/CombinedMinimumBuilder.h"
#include "Minuit/FunctionMinimum.h"
#include "Minuit/MnStrategy.h"
#include "Minuit/MnPrint.h"

FunctionMinimum CombinedMinimumBuilder::minimum(const MnFcn& fcn, const GradientCalculator& gc, const MinimumSeed& seed, const MnStrategy& strategy, unsigned int maxfcn, double edmval) const {

  FunctionMinimum min = theVMMinimizer.minimize(fcn, gc, seed, strategy, maxfcn, edmval);

  if(!min.isValid()) {
    std::cout<<"CombinedMinimumBuilder: migrad method fails, will try with simplex method first."<<std::endl; 
    MnStrategy str(2);
    FunctionMinimum min1 = theSimplexMinimizer.minimize(fcn, gc, seed, str, maxfcn, edmval);
    if(!min1.isValid()) {
      std::cout<<"CombinedMinimumBuilder: both migrad and simplex method fail."<<std::endl;
      return min1;
    }
    MinimumSeed seed1 = theVMMinimizer.seedGenerator()(fcn, gc, min1.userState(), str);
    
    FunctionMinimum min2 = theVMMinimizer.minimize(fcn, gc, seed1, str, maxfcn, edmval);
    if(!min2.isValid()) {
      std::cout<<"CombinedMinimumBuilder: both migrad and method fails also at 2nd attempt."<<std::endl;
      std::cout<<"CombinedMinimumBuilder: return simplex minimum."<<std::endl;
      return min1;
    }

    return min2;
  }

  return min;
}
