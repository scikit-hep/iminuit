#ifndef MN_SimplexMinimizer_H_
#define MN_SimplexMinimizer_H_

#include "Minuit/MnConfig.h"
#include "Minuit/ModularFunctionMinimizer.h"
#include "Minuit/SimplexBuilder.h"
#include "Minuit/SimplexSeedGenerator.h"

#include <vector>

class SimplexMinimizer : public ModularFunctionMinimizer {

public:

  SimplexMinimizer() : theSeedGenerator(SimplexSeedGenerator()), 
		       theBuilder(SimplexBuilder()) {}

  ~SimplexMinimizer() {}

  const MinimumSeedGenerator& seedGenerator() const {return theSeedGenerator;}
  const MinimumBuilder& builder() const {return theBuilder;}

private:

  SimplexSeedGenerator theSeedGenerator;
  SimplexBuilder theBuilder;
};

#endif //MN_SimplexMinimizer_H_
