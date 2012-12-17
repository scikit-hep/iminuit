#ifndef MN_FumiliBuilder_H_
#define MN_FumiliBuilder_H_

#include "Minuit/MinimumBuilder.h"
#include "Minuit/VariableMetricEDMEstimator.h"
#include "Minuit/FumiliErrorUpdator.h"
#include "Minuit/MnFcn.h"
#include "Minuit/FunctionMinimum.h"

/**

Builds the FunctionMinimum using the Fumili method.

@author Andras Zsenei, Creation date: 29 Sep 2004

@see <A HREF="http://www.cern.ch/winkler/minuit/tutorial/mntutorial.pdf">MINUIT Tutorial</A> on function minimization, section 5

@ingroup Minuit

\todo the role of the strategy in Fumili

*/



class FumiliBuilder : public MinimumBuilder {

public:

  FumiliBuilder() : theEstimator(VariableMetricEDMEstimator()), 
			    theErrorUpdator(FumiliErrorUpdator()) {}

  ~FumiliBuilder() {}


  /**

  Class the member function calculating the minimum and verifies the result
  depending on the strategy.

  @param theMnFcn the function to be minimized.

  @param theGradienCalculator not used in Fumili.

  @param theMinimumSeed the seed generator.

  @param theMnStrategy the strategy describing the number of function calls 
  allowed for gradient calculations.

  @param maxfcn maximum number of function calls after which the calculation 
  will be stopped even if it has not yet converged.

  @param edmval expected vertical distance to the minimum.

  @return Returns the function minimum found.


  \todo Complete the documentation by understanding what is the reason to 
  have two minimum methods.

  */

  virtual FunctionMinimum minimum(const MnFcn& theMnFcn, const GradientCalculator& theGradienCalculator, const MinimumSeed& theMinimumSeed, const MnStrategy& theMnStrategy, unsigned int maxfcn, double edmval) const;


  /**

  Calculates the minimum based on the Fumili method

  @param theMnFcn the function to be minimized.

  @param theGradienCalculator not used in Fumili

  @param theMinimumSeed the seed generator.

  @param states vector containing the state result of each iteration  

  @param maxfcn maximum number of function calls after which the calculation 
  will be stopped even if it has not yet converged.

  @param edmval expected vertical distance to the minimum

  @return Returns the function minimum found.

  @see <A HREF="http://www.cern.ch/winkler/minuit/tutorial/mntutorial.pdf">MINUIT Tutorial</A> on function minimization, section 5


  \todo some nice Latex based formula here...

  */

  FunctionMinimum minimum(const MnFcn& theMnFcn, const GradientCalculator& theGradienCalculator, const MinimumSeed& theMinimumSeed, std::vector<MinimumState> & states, unsigned int maxfcn, double edmval) const;
 

  /**

  Accessor to the EDM (expected vertical distance to the minimum) estimator.

  @return The EDM estimator used in the builder.

  \todo Maybe a little explanation concerning EDM in all relevant classes.

  */
 
  const VariableMetricEDMEstimator& estimator() const {return theEstimator;}


  /**

  Accessor to the error updator of the builder.

  @return The FumiliErrorUpdator used by the FumiliBuilder.

  */

  const FumiliErrorUpdator& errorUpdator() const {return theErrorUpdator;}


private:

  VariableMetricEDMEstimator theEstimator;
  FumiliErrorUpdator theErrorUpdator;
};

#endif //MN_FumiliBuilder_H_
