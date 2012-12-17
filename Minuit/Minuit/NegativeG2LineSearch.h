#ifndef MN_NegativeG2LineSearch_H_
#define MN_NegativeG2LineSearch_H_

class MnFcn;
class MinimumState;
class GradientCalculator;
class MnMachinePrecision;
class FunctionGradient;

/** In case that one of the components of the second derivative g2 calculated 
    by the numerical gradient calculator is negative, a 1dim line search in 
    the direction of that component is done in order to find a better position 
    where g2 is again positive. 
 */

class NegativeG2LineSearch {

public:

  NegativeG2LineSearch() {}
  
  ~NegativeG2LineSearch() {}

  MinimumState operator()(const MnFcn&, const MinimumState&, const  GradientCalculator&, const MnMachinePrecision&) const;

  bool hasNegativeG2(const FunctionGradient&, const MnMachinePrecision&) const;

private:

};

#endif //MN_NegativeG2LineSearch_H_
