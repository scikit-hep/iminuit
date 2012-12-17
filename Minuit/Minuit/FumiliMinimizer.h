#ifndef MN_FumiliMinimizer_H_
#define MN_FumiliMinimizer_H_

#include "Minuit/ModularFunctionMinimizer.h"
#include "Minuit/MnSeedGenerator.h"
#include "Minuit/FumiliBuilder.h"


class MinimumSeedGenerator;
class MinimumBuilder;
class MinimumSeed;
class MnFcn;
class FumiliFcnBase; 
class GradientCalculator;
class MnUserParameterState;
class MnUserParameters;
class MnUserCovariance;
class MnStrategy;




/** 

Instantiates the seed generator and minimum builder for the
Fumili minimization method. Produces the minimum via the 
minimize methods inherited from ModularFunctionMinimizer.

@author Andras Zsenei and Lorenzo Moneta, Creation date: 28 Sep 2004

@ingroup Minuit

*/


class FumiliMinimizer : public ModularFunctionMinimizer {

public:


  /**

  Constructor initializing the FumiliMinimizer by instantiatiating 
  the SeedGenerator and MinimumBuilder for the Fumili minimization method.

  @see MnSeedGenerator
  
  @see FumiliBuilder

  */

  FumiliMinimizer() : theMinSeedGen(MnSeedGenerator()),
			      theMinBuilder(FumiliBuilder()) {}
  
  ~FumiliMinimizer() {}


  /**

  Accessor to the seed generator of the minimizer.

  @return A reference to the seed generator used by the minimizer

  */

  const MinimumSeedGenerator& seedGenerator() const {return theMinSeedGen;}


  /**

  Accessor to the minimum builder of the minimizer.

  @return a reference to the minimum builder.

  */

  const FumiliBuilder& builder() const {return theMinBuilder;}


  // for Fumili

  FunctionMinimum minimize(const FCNBase&, const MnUserParameterState&, const MnStrategy&, unsigned int maxfcn = 0, double toler = 0.1) const;

  virtual FunctionMinimum minimize(const FCNGradientBase&, const MnUserParameterState&, const MnStrategy&, unsigned int maxfcn = 0, double toler = 0.1) const;

  // need to re-implement all function in ModularFuncitionMinimizer otherwise they will be hided

  virtual FunctionMinimum minimize(const FCNBase& fcn, const std::vector<double>& par, const std::vector<double>& err, unsigned int stra = 1, unsigned int maxfcn = 0, double toler = 0.1) const { 
    return ModularFunctionMinimizer::minimize(fcn, par, err, stra, maxfcn,toler);
  } 

  virtual FunctionMinimum minimize(const FCNGradientBase&fcn, const std::vector<double>&par, const std::vector<double>&err, unsigned int stra=1, unsigned int maxfcn = 0, double toler = 0.1) const { 
    return ModularFunctionMinimizer::minimize(fcn,par,err,stra,maxfcn,toler);    
  }

  virtual FunctionMinimum minimize(const FCNBase& fcn, const std::vector<double>&par, unsigned int nrow, const std::vector<double>&cov, unsigned int stra=1, unsigned int maxfcn = 0, double toler = 0.1) const { 
    return ModularFunctionMinimizer::minimize(fcn,par,nrow,cov,stra,maxfcn,toler);    
  } 

  virtual FunctionMinimum minimize(const FCNGradientBase& fcn, const std::vector<double>&par, unsigned int nrow, const std::vector<double>&cov, unsigned int stra=1, unsigned int maxfcn = 0, double toler = 0.1) const { 
    return ModularFunctionMinimizer::minimize(fcn,par,nrow,cov,stra,maxfcn,toler);    
  } 
 

  virtual FunctionMinimum minimize(const FCNBase& fcn, const MnUserParameters& par, const MnStrategy& stra, unsigned int maxfcn = 0, double toler = 0.1) const { 
    return ModularFunctionMinimizer::minimize(fcn,par,stra,maxfcn,toler); 
  }

  virtual FunctionMinimum minimize(const FCNGradientBase& fcn, const MnUserParameters& par, const MnStrategy& stra, unsigned int maxfcn = 0, double toler = 0.1) const { 
    return ModularFunctionMinimizer::minimize(fcn,par,stra,maxfcn,toler); 
  }

  virtual FunctionMinimum minimize(const FCNBase& fcn, const MnUserParameters& par, const MnUserCovariance& cov, const MnStrategy& stra, unsigned int maxfcn = 0, double toler = 0.1) const { 
    return ModularFunctionMinimizer::minimize(fcn,par,cov,stra,maxfcn,toler); 
  }

  virtual FunctionMinimum minimize(const FCNGradientBase& fcn, const MnUserParameters& par, const MnUserCovariance& cov, const MnStrategy& stra, unsigned int maxfcn = 0, double toler = 0.1) const { 
    return ModularFunctionMinimizer::minimize(fcn,par,cov,stra,maxfcn,toler); 
  }



  virtual FunctionMinimum minimize(const MnFcn& mfcn, const GradientCalculator& gc, const MinimumSeed& seed, const MnStrategy& stra, unsigned int maxfcn, double toler) const { 
    return ModularFunctionMinimizer::minimize(mfcn, gc, seed, stra, maxfcn, toler); 
  }


private:

  MnSeedGenerator theMinSeedGen;
  FumiliBuilder theMinBuilder;

};

#endif //MN_FumiliMinimizer_H_
