#ifndef MN_VariableMetricMinimizer_H_
#define MN_VariableMetricMinimizer_H_

#include "Minuit/MnConfig.h"
#include "Minuit/ModularFunctionMinimizer.h"
#include "Minuit/MnSeedGenerator.h"
#include "Minuit/VariableMetricBuilder.h"

/** Instantiates the SeedGenerator and MinimumBuilder for
    Variable Metric Minimization method.
 */

class VariableMetricMinimizer : public ModularFunctionMinimizer {

public:

  VariableMetricMinimizer() : theMinSeedGen(MnSeedGenerator()),
			      theMinBuilder(VariableMetricBuilder()) {}

  ~VariableMetricMinimizer() {}

  const MinimumSeedGenerator& seedGenerator() const {return theMinSeedGen;}
  const MinimumBuilder& builder() const {return theMinBuilder;}

private:

  MnSeedGenerator theMinSeedGen;
  VariableMetricBuilder theMinBuilder;
};

#endif //MN_VariableMetricMinimizer_H_
