
#include "Minuit/FumiliStandardChi2FCN.h"

#include <vector>
#include <cmath>

//#include <iostream>


std::vector<double> FumiliStandardChi2FCN::elements(const std::vector<double>& par) const {

    
  std::vector<double> result;
  double tmp1 = 0.0;
  unsigned int thePositionsSize = thePositions.size();
    
    
  for(unsigned int i=0; i < thePositionsSize; i++) {

      const std::vector<double> & currentPosition = thePositions[i];

      // The commented line is the object-oriented way to do it
      // but it is faster to do a single function call...
      //(*(this->getModelFunction())).setParameters(par);
      tmp1 = (*(this->modelFunction()))(par, currentPosition)- theMeasurements[i];
            
      result.push_back(tmp1*theInvErrors[i] );
    
      //std::cout << "element " << i << "  " << (*(this->getModelFunction()))(par, currentPosition) << "  " <<  theMeasurements[i] << "  " << result[i] << std::endl; 
  }



  return result;

}


 
const std::vector<double> & FumiliStandardChi2FCN::getMeasurement(int index) const {

  return thePositions[index];

}


int FumiliStandardChi2FCN::getNumberOfMeasurements() const {

  return thePositions.size();

}



void  FumiliStandardChi2FCN::evaluateAll( const std::vector<double> & par) { 

  // loop on the measurements 

  int nmeas = getNumberOfMeasurements(); 
  std::vector<double> & grad = gradient();
  std::vector<double> & h = hessian();
  int npar = par.size();
  double chi2 = 0; 
  grad.resize(npar);
  h.resize( static_cast<unsigned int>(0.5 * npar* (npar + 1) ) );
  // reset elements
  grad.assign(npar, 0.0);
  h.assign(static_cast<unsigned int>(0.5 * npar* (npar + 1) ) , 0.0);
 

  const ParametricFunction & modelFunc = *modelFunction();

  for (int i = 0; i < nmeas; ++i) { 

    // work for multi-dimensional points
    const std::vector<double> & currentPosition = thePositions[i];
    modelFunc.setParameters( currentPosition );
    double invError = theInvErrors[i];
    double fval = modelFunc(par); 
    
    double element = ( fval - theMeasurements[i] )*invError;
    chi2 += element*element;

    // calc derivatives 

    // this method should return a reference
    std::vector<double> mfg = modelFunc.getGradient(par);

    // grad is derivative of chi2 w.r.t to parameters
    for (int j = 0; j < npar ; ++j) { 
      double dfj = invError * mfg[j]; 
      grad[j] += 2.0 * element * dfj; 
      
      // in second derivative use Fumili approximation neglecting the term containing the 
      // second derivatives of the model function
      for (int k = j; k < npar; ++ k) { 
	int idx =  j + k*(k+1)/2; 
	h[idx] += 2.0 * dfj * invError * mfg[k]; 
      }
      
    } // end param loop  
  
  } // end points loop 

    // set value in base class
  setFCNValue( chi2); 

}
