#ifndef MN_VariableMetricBuilder_H_
#define MN_VariableMetricBuilder_H_

#include "Minuit/MnConfig.h"
#include "Minuit/MinimumBuilder.h"
#include "Minuit/VariableMetricEDMEstimator.h"
#include "Minuit/DavidonErrorUpdator.h"

#include <vector>

class VariableMetricBuilder : public MinimumBuilder {

public:

  VariableMetricBuilder() : theEstimator(VariableMetricEDMEstimator()), 
			    theErrorUpdator(DavidonErrorUpdator()) {}

  ~VariableMetricBuilder() {}

  virtual FunctionMinimum minimum(const MnFcn&, const GradientCalculator&, const MinimumSeed&, const MnStrategy&, unsigned int, double) const;

  FunctionMinimum minimum(const MnFcn&, const GradientCalculator&, const MinimumSeed&, std::vector<MinimumState> &, unsigned int, double) const;

  const VariableMetricEDMEstimator& estimator() const {return theEstimator;}
  const DavidonErrorUpdator& errorUpdator() const {return theErrorUpdator;}
  static int print_level;
  static void setPrintLevel(int p);
private:

  VariableMetricEDMEstimator theEstimator;
  DavidonErrorUpdator theErrorUpdator;
};

#endif //MN_VariableMetricBuilder_H_
