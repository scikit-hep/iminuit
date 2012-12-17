#ifndef MN_VariableMetricEDMEstimator_H_
#define MN_VariableMetricEDMEstimator_H_

class FunctionGradient;
class MinimumError;

class VariableMetricEDMEstimator {

public:

  VariableMetricEDMEstimator() {}
  
  ~VariableMetricEDMEstimator() {}

  double estimate(const FunctionGradient&, const MinimumError&) const;

private:

};

#endif //MN_VariableMetricEDMEstimator_H_
