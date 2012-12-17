#ifndef MN_ModularFunctionMinimizer_H_
#define MN_ModularFunctionMinimizer_H_

#include "Minuit/MnConfig.h"

#include "Minuit/FunctionMinimizer.h"
#include <vector>


class MinimumSeedGenerator;
class MinimumBuilder;
class MinimumSeed;
class MnFcn;
class GradientCalculator;
class MnUserParameterState;
class MnUserParameters;
class MnUserCovariance;
class MnStrategy;
class FumiliFCNBase;


class ModularFunctionMinimizer : public FunctionMinimizer {

public:

  virtual ~ModularFunctionMinimizer() {}

// inherited interface
  virtual FunctionMinimum minimize(const FCNBase&, const std::vector<double>&, const std::vector<double>&, unsigned int stra=1, unsigned int maxfcn = 0, double toler = 0.1) const; 

  virtual FunctionMinimum minimize(const FCNGradientBase&, const std::vector<double>&, const std::vector<double>&, unsigned int stra=1, unsigned int maxfcn = 0, double toler = 0.1) const; 

  virtual FunctionMinimum minimize(const FCNBase&, const std::vector<double>&, unsigned int, const std::vector<double>&, unsigned int stra=1, unsigned int maxfcn = 0, double toler = 0.1) const; 

  virtual FunctionMinimum minimize(const FCNGradientBase&, const std::vector<double>&, unsigned int, const std::vector<double>&, unsigned int stra=1, unsigned int maxfcn = 0, double toler = 0.1) const; 

// extension
  virtual FunctionMinimum minimize(const FCNBase&, const MnUserParameters&, const MnStrategy&, unsigned int maxfcn = 0, double toler = 0.1) const;

  virtual FunctionMinimum minimize(const FCNGradientBase&, const MnUserParameters&, const MnStrategy&, unsigned int maxfcn = 0, double toler = 0.1) const;

  virtual FunctionMinimum minimize(const FCNBase&, const MnUserParameters&, const MnUserCovariance&, const MnStrategy&, unsigned int maxfcn = 0, double toler = 0.1) const;

  virtual FunctionMinimum minimize(const FCNGradientBase&, const MnUserParameters&, const MnUserCovariance&, const MnStrategy&, unsigned int maxfcn = 0, double toler = 0.1) const;

  virtual FunctionMinimum minimize(const FCNBase&, const MnUserParameterState&, const MnStrategy&, unsigned int maxfcn = 0, double toler = 0.1) const;

  virtual FunctionMinimum minimize(const FCNGradientBase&, const MnUserParameterState&, const MnStrategy&, unsigned int maxfcn = 0, double toler = 0.1) const;

  // for Fumili

//   virtual FunctionMinimum minimize(const FumiliFCNBase&, const std::vector<double>&, const std::vector<double>&, unsigned int stra=1, unsigned int maxfcn = 0, double toler = 0.1) const; 

//   virtual FunctionMinimum minimize(const FumiliFCNBase&, const MnUserParameters&, const MnStrategy&, unsigned int maxfcn = 0, double toler = 0.1) const;

//   virtual FunctionMinimum minimize(const FumiliFCNBase&, const MnUserParameters&, const MnUserCovariance&, const MnStrategy&, unsigned int maxfcn = 0, double toler = 0.1) const;

//   virtual FunctionMinimum minimize(const FumiliFCNBase&, const MnUserParameterState&, const MnStrategy&, unsigned int maxfcn = 0, double toler = 0.1) const;


  virtual const MinimumSeedGenerator& seedGenerator() const = 0;
  virtual const MinimumBuilder& builder() const = 0;

public:

  virtual FunctionMinimum minimize(const MnFcn&, const GradientCalculator&, const MinimumSeed&, const MnStrategy&, unsigned int, double) const; 

};

#endif //MN_ModularFunctionMinimizer_H_
