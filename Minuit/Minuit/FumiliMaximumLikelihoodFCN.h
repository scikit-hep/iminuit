#ifndef MN_FumiliMaximumLikelihoodFCN_H_
#define MN_FumiliMaximumLikelihoodFCN_H_

#include "FumiliFCNBase.h"
#include <vector>
#include <cmath>
#include <float.h>

//#include <iostream>

/** 

Extension of the FCNBase for the Fumili method. Fumili applies only to 
minimization problems used for fitting. The method is based on a 
linearization of the model function negleting second derivatives. 
User needs to provide the model function. In this cased the function
to be minimized is the sum of the logarithms of the model function
for the different measurements times -1.


@author Andras Zsenei and Lorenzo Moneta, Creation date: 3 Sep 2004

@see <A HREF="http://www.cern.ch/winkler/minuit/tutorial/mntutorial.pdf">MINUIT Tutorial</A> on function minimization, section 5

@see FumiliStandardMaximumLikelihoodFCN

@ingroup Minuit

\todo Insert a nice latex formula...

*/



class FumiliMaximumLikelihoodFCN : public FumiliFCNBase {

public:

  FumiliMaximumLikelihoodFCN() {}

  virtual ~FumiliMaximumLikelihoodFCN() {}


  /**

  Sets the model function for the data (for example gaussian+linear for a peak)

  @param modelFunction a reference to the model function.

  */

  void setModelFunction(const ParametricFunction& modelFCN) { theModelFunction = &modelFCN; }



  /**

  Returns the model function used for the data.

  @return Returns a pointer to the model function.

  */

  const ParametricFunction*  modelFunction() const { return theModelFunction; }



  /**
     
  Evaluates the model function for the different measurement points and 
  the parameter values supplied, calculates a figure-of-merit for each
  measurement and returns a vector containing the result of this
  evaluation.

  @param par vector of parameter values to feed to the model function.

  @return A vector containing the figures-of-merit for the model function evaluated 
  for each set of measurements.

  */

  virtual std::vector<double> elements(const std::vector<double>& par) const = 0;



  /**
     
  Accessor to the parameters of a given measurement. For example in the
  case of a chi-square fit with a one-dimensional Gaussian, the parameter 
  characterizing the measurement will be the position. It is the parameter
  that is feeded to the model function.

  @param index index of the measueremnt the parameters of which to return
  @return A vector containing the values characterizing a measurement

  */

  virtual const std::vector<double> & getMeasurement(int index) const = 0;


  /**

  Accessor to the number of measurements used for calculating the 
  present figure of merit.

  @return the number of measurements

  */

  virtual int getNumberOfMeasurements() const = 0;


  /**
 
  Calculates the function for the maximum likelihood method. The user must 
  implement in a class which inherits from FumiliChi2FCN the member function
  elements() which will supply the elements for the sum.


  @param par vector containing the parameter values for the model function
  
  @return The sum of the natural logarithm of the elements multiplied by -1

  @see FumiliFCNBase#elements

  */
  
  double operator()(const std::vector<double>& par) const {

    double sumoflogs = 0.0; 
    std::vector<double> vecElements =  elements(par);
    unsigned int vecElementsSize = vecElements.size();

    for (unsigned int i = 0; i < vecElementsSize; ++i) { 
      double tmp = vecElements[i]; 
      //for max likelihood probability have to be positive
      assert(tmp >= 0);
      if ( tmp < FLT_MIN*5 )
	tmp = FLT_MIN*5; 

      sumoflogs -= std::log(tmp);
      //std::cout << " i " << tmp << " lik " << sumoflogs << std::endl;
    }
      

    return sumoflogs; 
  }
  


  /**
     
  !!!!!!!!!!!! to be commented

  */

  virtual double up() const { return 0.5; }  
   
 private: 

  // A pointer to the model function which describes the data
  const ParametricFunction *theModelFunction;

};

#endif //MN_FumiliMaximumLikelihoodFCN_H_
