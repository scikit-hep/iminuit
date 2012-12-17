#include "Minuit/MnStrategy.h"

//default strategy
MnStrategy::MnStrategy() {
  setMediumStrategy();
}

//user defined strategy (0, 1, >=2)
MnStrategy::MnStrategy(unsigned int stra) {
  if(stra == 0) setLowStrategy();
  else if(stra == 1) setMediumStrategy();
  else setHighStrategy();
}

void MnStrategy::setLowStrategy() {
  theStrategy = 0;
  setGradientNCycles(2);
  setGradientStepTolerance(0.5);
  setGradientTolerance(0.1);
  setHessianNCycles(3);
  setHessianStepTolerance(0.5);
  setHessianG2Tolerance(0.1);
  setHessianGradientNCycles(1);
}

void MnStrategy::setMediumStrategy() {
  theStrategy = 1;
  setGradientNCycles(3);
  setGradientStepTolerance(0.3);
  setGradientTolerance(0.05);
  setHessianNCycles(5);
  setHessianStepTolerance(0.3);
  setHessianG2Tolerance(0.05);
  setHessianGradientNCycles(2);
}

void MnStrategy::setHighStrategy() {
  theStrategy = 2;
  setGradientNCycles(5);
  setGradientStepTolerance(0.1);
  setGradientTolerance(0.02);
  setHessianNCycles(7);
  setHessianStepTolerance(0.1);
  setHessianG2Tolerance(0.02);
  setHessianGradientNCycles(6);
}
