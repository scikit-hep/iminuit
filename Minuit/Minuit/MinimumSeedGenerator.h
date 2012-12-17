#ifndef MN_MinimumSeedGenerator_H_
#define MN_MinimumSeedGenerator_H_

class MinimumSeed;
class MnFcn;
class GradientCalculator;
class MnUserParameterState;
class MnStrategy;
class AnalyticalGradientCalculator;

/** base class for seed generators (starting values); the seed generator 
    prepares initial starting values from the input (MnUserParameterState)
    for the minimization;
 */

class MinimumSeedGenerator {

public:

  virtual ~MinimumSeedGenerator() {}

  virtual MinimumSeed operator()(const MnFcn&, const GradientCalculator&, const MnUserParameterState&, const MnStrategy&) const = 0;

  virtual MinimumSeed operator()(const MnFcn&, const AnalyticalGradientCalculator&, const MnUserParameterState&, const MnStrategy&) const = 0;
};

#endif //MN_MinimumSeedGenerator_H_
