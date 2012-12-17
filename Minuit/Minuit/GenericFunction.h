#ifndef MN_GenericFunction_H_
#define MN_GenericFunction_H_

#include "Minuit/MnConfig.h"

#include <vector>


/** 

Class from which all the other classes, representing functions,
inherit. That is why it defines only one method, the operator(),
which allows to call the function.

@author Andras Zsenei and Lorenzo Moneta, Creation date: 23 Sep 2004

@ingroup Minuit

 */

class GenericFunction {

public:

  virtual ~GenericFunction() {}


  /**

  Evaluates the function using the vector containing the input values.

  @param x vector of the coordinates (for example the x coordinate for a 
  one-dimensional Gaussian)

  @return the result of the evaluation of the function.

  */

  virtual double operator()(const std::vector<double>& x) const=0;



};

#endif //MN_GenericFunction_H_
