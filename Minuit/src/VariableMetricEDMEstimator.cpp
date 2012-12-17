#include "Minuit/VariableMetricEDMEstimator.h"
#include "Minuit/FunctionGradient.h"
#include "Minuit/MinimumError.h"

double similarity(const LAVector&, const LASymMatrix&);

double VariableMetricEDMEstimator::estimate(const FunctionGradient& g, const MinimumError& e) const {

  if(e.invHessian().size()  == 1) 
    return 0.5*g.grad()(0)*g.grad()(0)*e.invHessian()(0,0);

  double rho = similarity(g.grad(), e.invHessian());
  return 0.5*rho;
}
