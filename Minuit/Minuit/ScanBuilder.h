#ifndef MN_ScanBuilder_H_
#define MN_ScanBuilder_H_

#include "Minuit/MinimumBuilder.h"

class FunctionMinimum;
class MnFcn;
class MinimumSeed;

/** Performs a minimization using the simplex method of Nelder and Mead
    (ref. Comp. J. 7, 308 (1965)).
 */

class ScanBuilder : public MinimumBuilder {

public:

  ScanBuilder() {}

  ~ScanBuilder() {}

  virtual FunctionMinimum minimum(const MnFcn&, const GradientCalculator&, const MinimumSeed&, const MnStrategy&, unsigned int, double) const;

private:

};

#endif //MN_ScanBuilder_H_
