#ifndef MN_CombinedMinimumBuilder_H_
#define MN_CombinedMinimumBuilder_H_

#include "Minuit/MinimumBuilder.h"
#include "Minuit/VariableMetricMinimizer.h"
#include "Minuit/SimplexMinimizer.h"

class CombinedMinimumBuilder : public MinimumBuilder {

public:

  CombinedMinimumBuilder() : theVMMinimizer(VariableMetricMinimizer()), 
			     theSimplexMinimizer(SimplexMinimizer()) {}

  ~CombinedMinimumBuilder() {}

  virtual FunctionMinimum minimum(const MnFcn&, const GradientCalculator&, const MinimumSeed&, const MnStrategy&, unsigned int, double) const;

private:

  VariableMetricMinimizer theVMMinimizer;
  SimplexMinimizer theSimplexMinimizer;
};

#endif //MN_CombinedMinimumBuilder_H_
