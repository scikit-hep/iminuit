#ifndef MN_ParametricFunction_H_
#define MN_ParametricFunction_H_

#include "Minuit/MnConfig.h"
#include <vector>
#include <cassert>

#include "Minuit/FCNBase.h"


/** 

Function which has parameters. For example, one could define
a one-dimensional Gaussian, by considering x as an input coordinate
for the evaluation of the function, and the mean and the square root
of the variance as parameters.
<p>
AS OF NOW PARAMETRICFUNCTION INHERITS FROM FCNBASE INSTEAD OF 
GENERICFUNCTION. THIS IS ONLY BECAUSE NUMERICAL2PGRADIENTCALCULATOR
NEEDS AN FCNBASE OBJECT AND WILL BE CHANGED!!!!!!!!!!!!!!!!

@ingroup Minuit

\todo ParametricFunction and all the classes that inherit from it 
are inheriting also FCNBase so that the gradient calculation has
the up() member function. That is not really good...


 */

class ParametricFunction : public FCNBase {

public:


  /**

  Constructor which initializes the ParametricFunction with the 
  parameters given as input.

  @param params vector containing the initial parameter values

  */
  
  ParametricFunction(const std::vector<double>& params) : par(params) {}



  /**

  Constructor which initializes the ParametricFunction by setting
  the number of parameters.

  @param nparams number of parameters of the parametric function

  */

  ParametricFunction(int nparams) : par(nparams) {}



  virtual ~ParametricFunction() {}



  /**

  Sets the parameters of the ParametricFunction.

  @param params vector containing the parameter values

  */

  virtual void setParameters(const std::vector<double>& params) const {

    assert(params.size() == par.size());
    par = params;

  }



  /**

  Accessor for the state of the parameters.

  @return vector containing the present parameter settings

  */

  virtual const std::vector<double> & getParameters() const { return par; }




  /**

  Accessor for the number of  parameters.

  @return the number of function parameters

  */
  virtual unsigned int  numberOfParameters() const { return par.size(); }

  // Why do I need to declare it here, it should be inherited without
  // any problems, no?

  /**

  Evaluates the function with the given coordinates.

  @param x vector containing the input coordinates

  @return the result of the function evaluation with the given
  coordinates.

  */

  virtual double operator()(const std::vector<double>& x) const=0;



  /**

  Evaluates the function with the given coordinates and parameter
  values. This member function is useful to implement when speed
  is an issue as it is faster to call only one function instead
  of two (setParameters and operator()). The default implementation,
  provided for convenience, does the latter.

  @param x vector containing the input coordinates

  @param params vector containing the parameter values

  @return the result of the function evaluation with the given
  coordinates and parameters

  */

  virtual double operator()(const std::vector<double>& x, const std::vector<double>& params) const {
    setParameters(params);
    return operator()(x);

  }


  /**

  Member function returning the gradient of the function with respect
  to its variables (but without including gradients with respect to
  its internal parameters).

  @param x vector containing the coordinates of the point where the
  gradient is to be calculated.

  @return the gradient vector of the function at the given point.

  */

  virtual std::vector<double>  getGradient(const std::vector<double>& x) const;




 protected:

  /**
  
  The vector containing the parameters of the function
  It is mutable for "historical reasons" as in the hierarchy
  methods and classes are const and all the implications of changing
  them back to non-const are not clear.

  */


  mutable std::vector<double> par;

};
 
#endif //MN_ParametricFunction_H_
