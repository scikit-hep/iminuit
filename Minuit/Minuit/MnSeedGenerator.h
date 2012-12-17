#ifndef MN_MnSeedGenerator_H_
#define MN_MnSeedGenerator_H_

#include "Minuit/MinimumSeedGenerator.h"

/** concrete implementation of the MinimumSeedGenerator interface; used within
    ModularFunctionMinimizer;
 */

class MnSeedGenerator : public MinimumSeedGenerator {

public:

  MnSeedGenerator() {}

  virtual ~MnSeedGenerator() {}

  virtual MinimumSeed operator()(const MnFcn&, const GradientCalculator&, const MnUserParameterState&, const MnStrategy&) const;

  virtual MinimumSeed operator()(const MnFcn&, const AnalyticalGradientCalculator&, const MnUserParameterState&, const MnStrategy&) const;

private:

};

#endif //MN_MnSeedGenerator_H_
