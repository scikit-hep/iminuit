#ifndef MN_SimplexSeedGenerator_H_
#define MN_SimplexSeedGenerator_H_

#include "Minuit/MinimumSeedGenerator.h"

class MinimumSeed;
class MnFcn;
class MnUserParameterState;
class MnStrategy;

class SimplexSeedGenerator : public MinimumSeedGenerator {

public:

  SimplexSeedGenerator() {}

  ~SimplexSeedGenerator() {}

  virtual MinimumSeed operator()(const MnFcn&, const GradientCalculator&, const MnUserParameterState&, const MnStrategy&) const;

  virtual MinimumSeed operator()(const MnFcn&, const AnalyticalGradientCalculator&, const MnUserParameterState&, const MnStrategy&) const;

private:

};

#endif //MN_SimplexSeedGenerator_H_
