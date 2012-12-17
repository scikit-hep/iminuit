#ifndef MN_FumiliStandardChi2FCN_H_
#define MN_FumiliStandardChi2FCN_H_

 
#include "Minuit/FumiliChi2FCN.h"
#include "Minuit/ParametricFunction.h"
#include <assert.h>
#include <vector>
#include <cmath>


/**

Class implementing the standard chi square function, which
is the sum of the squares of the figures-of-merit calculated for each measurement 
point, the individual figures-of-merit being: (the value predicted by the 
model-measured value)/standard deviation.

@author Andras Zsenei and Lorenzo Moneta, Creation date: 31 Aug 2004

@see FumiliChi2FCN

@ingroup Minuit

\todo nice formula for the documentation...

*/

class FumiliStandardChi2FCN : public FumiliChi2FCN {

public:


  /**
     
  Constructor which initializes chi square function for one-dimensional model function
  
  @param modelFCN the model function used for describing the data.

  @param meas vector containing the measured values.

  @param pos vector containing the x values corresponding to the
  measurements

  @param mvar vector containing the variances corresponding to each 
  measurement (where the variance equals the standard deviation squared).
  If the variances are zero, a value of 1 is used (as it is done in ROOT/PAW)

  */

  FumiliStandardChi2FCN(const ParametricFunction& modelFCN, const std::vector<double>& meas,
	   const std::vector<double>& pos,
	   const std::vector<double>& mvar) : 
    theErrorDef(1.) { //this->theModelFCN = &modelFunction; 
    this->setModelFunction(modelFCN); 

    assert(meas.size() == pos.size());
    assert(meas.size() == mvar.size());
    theMeasurements = meas;
    std::vector<double> x(1); 
    unsigned int n = mvar.size(); 
    thePositions.reserve( n);
    // correct for variance == 0
    theInvErrors.resize(n);
    for (unsigned int i = 0; i < n; ++i)  
    { 
      x[0] = pos[i];
      thePositions.push_back(x);
      // PAW/ROOT hack : use 1 for 0 entries bins
      if (mvar[i] == 0) 
	theInvErrors[i] = 1; 
      else 
	theInvErrors[i] = 1.0/sqrt(mvar[i]); 
    }

  }


  /**
     
  Constructor which initializes the multi-dimensional model function.
  
  @param modelFCN the model function used for describing the data.

  @param meas vector containing the measured values.

  @param pos vector containing the x values corresponding to the
  measurements

  @param mvar vector containing the variances corresponding to each 
  measurement (where the variance equals the standard deviation squared).
  If the variances are zero, a value of 1 is used (as it is done in ROOT/PAW)

  */

  FumiliStandardChi2FCN(const ParametricFunction& modelFCN, const std::vector<double>& meas,
	   const std::vector<std::vector<double> >& pos,
	   const std::vector<double>& mvar) : 
    theErrorDef(1.) { //this->theModelFCN = &modelFunction; 
    this->setModelFunction(modelFCN); 

    assert(meas.size() == pos.size());
    assert(meas.size() == mvar.size());
    theMeasurements = meas;
    thePositions = pos;
    // correct for variance == 0
    unsigned int n = mvar.size(); 
    theInvErrors.resize(n);
    for (unsigned int i = 0; i < n; ++i)  
    { 
      // PAW/ROOT hack : use 1 for 0 entries bins
      if (mvar[i] == 0) 
	theInvErrors[i] = 1; 
      else 
	theInvErrors[i] = 1.0/sqrt(mvar[i]); 
    }

  }




  ~FumiliStandardChi2FCN() {}





  /**

  Evaluates the model function for the different measurement points and 
  the parameter values supplied, calculates a figure-of-merit for each
  measurement and returns a vector containing the result of this
  evaluation. The figure-of-merit is (value predicted by the model 
  function-measured value)/standard deviation.

  @param par vector of parameter values to feed to the model function.

  @return A vector containing the figures-of-merit for the model function evaluated 
  for each set of measurements.  

  \todo What to do when the variances are 0???!! (right now just pushes back 0...)

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
  the chi-square.

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


  std::vector<double> theMeasurements;
  // support multi dim coordinates
  std::vector<std::vector<double> > thePositions;
  std::vector<double> theInvErrors;
  double theErrorDef;
  
 

};

#endif //MN_FumiliStandardChi2FCN_H_
