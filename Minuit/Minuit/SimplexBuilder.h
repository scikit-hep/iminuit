#ifndef MN_SimplexBuilder_H_
#define MN_SimplexBuilder_H_

#include "Minuit/MinimumBuilder.h"

class FunctionMinimum;
class MnFcn;
class MinimumSeed;

/** Performs a minimization using the simplex method of Nelder and Mead
    (ref. Comp. J. 7, 308 (1965)).
 */

class SimplexBuilder : public MinimumBuilder {

public:

  SimplexBuilder() {}

  ~SimplexBuilder() {}

  virtual FunctionMinimum minimum(const MnFcn&, const GradientCalculator&, const MinimumSeed&, const MnStrategy&, unsigned int, double) const;

private:

};

#endif //MN_SimplexBuilder_H_
