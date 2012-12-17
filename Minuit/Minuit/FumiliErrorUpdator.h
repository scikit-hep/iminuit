#ifndef MN_FumiliErrorUpdator_H_
#define MN_FumiliErrorUpdator_H_

#include "Minuit/MinimumErrorUpdator.h"

class MinimumState; 
class MinimumParameters; 
class GradientCalculator; 
class FumiliFCNBase; 
class FunctionGradient; 

/**

In the case of the Fumili algorithm the error matrix (or the Hessian
matrix containing the (approximate) second derivatives) is calculated
using a linearization of the model function negleting second 
derivatives. (In some sense the name Updator is a little bit misleading
as the error matrix is not calculated by iteratively updating, like
in Davidon's or other similar variable metric methods, but by
recalculating each time).


@author  Andras Zsenei and Lorenzo Moneta, Creation date: 28 Sep 2004

@see <A HREF="http://www.cern.ch/winkler/minuit/tutorial/mntutorial.pdf">MINUIT Tutorial</A> on function minimization, section 5

@see DavidonErrorUpdator

@ingroup Minuit

*/

class FumiliErrorUpdator : public MinimumErrorUpdator {

public:

  FumiliErrorUpdator() {}
  
  ~FumiliErrorUpdator() {  }



  /**
     
  Member function that calculates the error matrix (or the Hessian
  matrix containing the (approximate) second derivatives) using a 
  linearization of the model function negleting second derivatives.

  @param theMinimumState used to calculate the change in the covariance
  matrix between the two iterations

  @param theMinimumParameters the parameters at the present iteration

  @param theGradientCalculator the gradient calculator used to retrieved the parameter transformation

  @param theFumiliFCNBase the function calculating the figure of merit.
  

  \todo Some nice latex mathematical formuli...

  */

  virtual MinimumError update(const MinimumState& theMinimumState, 
			      const MinimumParameters& theMinimumParameters,
			      const GradientCalculator& theGradientCalculator, 
			      double lambda) const;



  /**

  Member function which is only present due to the design already in place
  of the software. As all classes calculating the error matrix are supposed
  inherit from the MinimumErrorUpdator they must inherit this method. In some 
  methods calculating the aforementioned matrix some of these parameters are
  not needed and other parameters are necessary... Hopefully, a more elegant
  solution will be found in the future.

  \todo How to get rid of this dummy method which is only due to the inheritance

  */

  virtual MinimumError update(const MinimumState&, const MinimumParameters&,
			      const FunctionGradient&) const;



private:


};

#endif //MN_FumiliErrorUpdator_H_
