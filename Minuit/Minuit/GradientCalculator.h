#ifndef MN_GradientCalculator_H_
#define MN_GradientCalculator_H_

class MinimumParameters;
class FunctionGradient;

class GradientCalculator {

public:
  
  virtual ~GradientCalculator() {}

  virtual FunctionGradient operator()(const MinimumParameters&) const = 0;

  virtual FunctionGradient operator()(const MinimumParameters&,
				      const FunctionGradient&) const = 0;
};

#endif //MN_GradientCalculator_H_
