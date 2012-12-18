#ifndef MN_MinimumBuilder_H_
#define MN_MinimumBuilder_H_

class FunctionMinimum;
class MnFcn;
class GradientCalculator;
class MinimumSeed;
class MnStrategy;

class MinimumBuilder {

public:

  virtual ~MinimumBuilder() {}

  virtual FunctionMinimum minimum(const MnFcn&, const GradientCalculator&, const MinimumSeed&, const MnStrategy&, unsigned int, double) const = 0;

};

#endif //MN_MinimumBuilder_H_
