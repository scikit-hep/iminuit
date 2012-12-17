#ifndef MN_FCNBase_H_
#define MN_FCNBase_H_

#include "Minuit/MnConfig.h"

#include <vector>

#include "Minuit/GenericFunction.h"

/**

@defgroup Minuit Minuit Math Library

*/


/** 


The function to be minimized, which has to be implemented by the user.

@author Fred James and Matthias Winkler; modified by Andras Zsenei and Lorenzo Moneta

@ingroup Minuit

 */

class FCNBase : public GenericFunction {

public:


  virtual ~FCNBase() {}



  /**

  The meaning of the vector of parameters is of course defined by the user, 
  who uses the values of those parameters to calculate his function value. 
  The order and the position of these parameters is strictly the one specified 
  by the user when supplying the starting values for minimization. The starting 
  values must be specified by the user, either via an std::vector<double> or the 
  MnUserParameters supplied as input to the MINUIT minimizers such as 
  VariableMetricMinimizer or MnMigrad. Later values are determined by MINUIT 
  as it searches for the minimum or performs whatever analysis is requested by 
  the user.

  @param par function parameters as defined by the user.

  @return the value of the function.

  @see MnUserParameters
  @see VariableMetricMinimizer 
  @see MnMigrad

  */

  virtual double operator()(const std::vector<double>& x) const = 0;

  

  /**

  Error definition of the function. MINUIT defines parameter errors as the 
  change in parameter value required to change the function value by up. Normally, 
  for chisquared fits it is 1, and for negative log likelihood, its value is 0.5.
  If the user wants instead the 2-sigma errors for chisquared fits, it becomes 4, 
  as Chi2(x+n*sigma) = Chi2(x) + n*n.
  
  Comment a little bit better with links!!!!!!!!!!!!!!!!!

  */

  virtual double errorDef() const {return up();}


  /**

  Error definition of the function. MINUIT defines parameter errors as the 
  change in parameter value required to change the function value by up. Normally, 
  for chisquared fits it is 1, and for negative log likelihood, its value is 0.5.
  If the user wants instead the 2-sigma errors for chisquared fits, it becomes 4, 
  as Chi2(x+n*sigma) = Chi2(x) + n*n.
    
  \todo Comment a little bit better with links!!!!!!!!!!!!!!!!! Idem for errorDef()

  */

  virtual double up() const = 0;




};

#endif //MN_FCNBase_H_
