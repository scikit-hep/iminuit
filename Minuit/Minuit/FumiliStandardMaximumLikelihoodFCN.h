#ifndef MN_FumiliStandardMaximumLikelihoodFCN_H_
#define MN_FumiliStandardMaximumLikelihoodFCN_H_


#include "Minuit/FumiliMaximumLikelihoodFCN.h"
#include "Minuit/ParametricFunction.h"
#include <vector>



/**

Class implementing the elements member function for the standard 
maximum likelihood method.

@author Andras Zsenei and Lorenzo Moneta, Creation date: 4 Sep 2004

@see FumiliMaximumLikelihoodFCN

@ingroup Minuit

*/

class FumiliStandardMaximumLikelihoodFCN : public FumiliMaximumLikelihoodFCN {

public:


  /**
     
  Constructor which initializes the measurement points for the one dimensional model function.
  
  @param modelFCN the model function used for describing the data.

  @param pos vector containing the x values corresponding to the
  measurements

  */

  FumiliStandardMaximumLikelihoodFCN(const ParametricFunction& modelFCN, 
				     const std::vector<double>& pos) : 
    theErrorDef(0.5) 
  {
    this->setModelFunction(modelFCN); 
    unsigned int n = pos.size(); 
    thePositions.reserve( n );
    std::vector<double> x(1);
    for (unsigned int i = 0; i < n; ++i) { 
      x[0] = pos[i];
      thePositions.push_back(x);  
    }
  }



  /**
     
  Constructor which initializes the measurement points for the multi dimensional model function.
  
  @param modelFCN the model function used for describing the data.

  @param pos vector containing the x values corresponding to the
  measurements

  */

  FumiliStandardMaximumLikelihoodFCN(const ParametricFunction& modelFCN, 
				     const std::vector<std::vector<double> >& pos) : 
    theErrorDef(0.5) {
    this->setModelFunction(modelFCN); 
    thePositions = pos;

  }




  ~FumiliStandardMaximumLikelihoodFCN() {}




  /**

  Evaluates the model function for the different measurement points and 
  the parameter values supplied.

  @param par vector of parameter values to feed to the model function.

  @return A vector containing the model function evaluated 
  for each measurement point.  

  */

  std::vector<double> elements(const std::vector<double>& par) const;




  /**

  Accessor to the position of the measurement (x coordinate).

  @param index index of the measuerement the position of which to return.

  @return the position of the measurement.

  */

  virtual const std::vector<double> & getMeasurement(int index) const;


  /**

  Accessor to the number of measurements used for calculating 
  the maximum likelihood.

  @return the number of measurements.

  */

  virtual int getNumberOfMeasurements() const;

  /**
  
  Evaluate function value, gradient and hessian using Fumili approximation, for values of parameters p
  The resul is cached inside and is return from the FumiliFCNBase::value ,  FumiliFCNBase::gradient and 
  FumiliFCNBase::hessian methods 

  @param par vector of parameters

  **/

  virtual  void evaluateAll( const std::vector<double> & par );


 private:

  std::vector<std::vector<double> > thePositions;
  double theErrorDef;
  
    

};

#endif //MN_FumiliStandardMaximumLikelihoodFCN_H_
