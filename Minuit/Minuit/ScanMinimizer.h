#ifndef MN_ScanMinimizer_H_
#define MN_ScanMinimizer_H_

#include "Minuit/MnConfig.h"
#include "Minuit/ModularFunctionMinimizer.h"
#include "Minuit/ScanBuilder.h"
#include "Minuit/SimplexSeedGenerator.h"

#include <vector>

class ScanMinimizer : public ModularFunctionMinimizer {

public:

  ScanMinimizer() : theSeedGenerator(SimplexSeedGenerator()), 
		    theBuilder(ScanBuilder()) {}
  
  ~ScanMinimizer() {}
  
  const MinimumSeedGenerator& seedGenerator() const {return theSeedGenerator;}
  const MinimumBuilder& builder() const {return theBuilder;}
  
private:
  
  SimplexSeedGenerator theSeedGenerator;
  ScanBuilder theBuilder;
};

#endif //MN_ScanMinimizer_H_
