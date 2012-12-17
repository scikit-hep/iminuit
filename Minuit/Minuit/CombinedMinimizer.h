#ifndef MN_CombinedMinimizer_H_
#define MN_CombinedMinimizer_H_

#include "Minuit/ModularFunctionMinimizer.h"
#include "Minuit/MnSeedGenerator.h"
#include "Minuit/CombinedMinimumBuilder.h"

/** Combined minimizer: if migrad method fails at first attempt, a simplex
    minimization is performed and then migrad is tried again.
 */

class CombinedMinimizer : public ModularFunctionMinimizer {

public:

  CombinedMinimizer() : theMinSeedGen(MnSeedGenerator()),
			theMinBuilder(CombinedMinimumBuilder()) {}
  
  ~CombinedMinimizer() {}

  const MinimumSeedGenerator& seedGenerator() const {return theMinSeedGen;}
  const MinimumBuilder& builder() const {return theMinBuilder;}

private:

  MnSeedGenerator theMinSeedGen;
  CombinedMinimumBuilder theMinBuilder;
};

#endif //MN_CombinedMinimizer_H_
