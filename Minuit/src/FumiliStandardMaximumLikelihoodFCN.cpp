
#include "Minuit/FumiliStandardMaximumLikelihoodFCN.h"

#include <vector>
#include <cmath>
#include <float.h>

//#include <iostream>

  std::vector<double> FumiliStandardMaximumLikelihoodFCN::elements(const std::vector<double>& par) const {

    
    std::vector<double> result;
    double tmp1 = 0.0;
    unsigned int thePositionsSize = thePositions.size();

    
    for(unsigned int i=0; i < thePositionsSize; i++) {

      const std::vector<double> & currentPosition = thePositions[i];

      // The commented line is the object-oriented way to do it
      // but it is faster to do a single function call...
      //(*(this->getModelFunction())).setParameters(par);
      tmp1 = (*(this->modelFunction()))(par, currentPosition);

      // std::cout << " i = " << i << "  " << currentPosition[0] << "  " << tmp1 << std::endl; 
            
      result.push_back(tmp1);

    }


    return result;

  }



const std::vector<double> & FumiliStandardMaximumLikelihoodFCN::getMeasurement(int index) const {

  return thePositions[index];

}


int FumiliStandardMaximumLikelihoodFCN::getNumberOfMeasurements() const {

  return thePositions.size();

}


void  FumiliStandardMaximumLikelihoodFCN::evaluateAll( const std::vector<double> & par) { 

  static double minDouble = 8.0*DBL_MIN; 
  static double minDouble2 = sqrt(8.0*DBL_MIN); 
  static double maxDouble2 = 1.0/minDouble2;
  // loop on the measurements 

  int nmeas = getNumberOfMeasurements(); 
  std::vector<double> & grad = gradient();
  std::vector<double> & h = hessian();
  int npar = par.size();
  double logLikelihood = 0; 
  grad.resize(npar);
  h.resize( static_cast<unsigned int>(0.5 * npar* (npar + 1) ) );
  grad.assign(npar, 0.0);
  h.assign(static_cast<unsigned int>(0.5 * npar* (npar + 1) ) , 0.0);

  const ParametricFunction & modelFunc = *modelFunction();

  for (int i = 0; i < nmeas; ++i) { 

    // work for one-dimensional points
    const std::vector<double> & currentPosition = thePositions[i];
    modelFunc.setParameters( currentPosition );
    double fval = modelFunc(par); 
    if (fval < minDouble ) fval = minDouble;   // to avoid getting infinity and nan's
    logLikelihood -= std::log( fval); 
    double invFval = 1.0/fval;
    // this method should return a reference
    std::vector<double> mfg = modelFunc.getGradient(par);

    // calc derivatives 

    for (int j = 0; j < npar; ++j) { 
      if ( fabs(mfg[j]) < minDouble ) {
// 	std::cout << "SMALL values: grad =  " << mfg[j] << "  "  << minDouble << " f(x) = " << fval 
// 		  << " params " << j << " p0 = " << par[0] << " p1 = " << par[1] <<  std::endl;
	if (mfg[j] < 0)  
	  mfg[j] =  -minDouble;
	else 
	  mfg[j] =  minDouble;
      }	  

      double dfj = invFval * mfg[j]; 
      // to avoid summing infinite and nan later when calculating the hessian
      if ( fabs(dfj) > maxDouble2 ) { 
	if (dfj > 0) 
	  dfj = maxDouble2; 
	else 
	  dfj = -maxDouble2;
      }

      grad[j] -= dfj;
//       if ( ! ( dfj > 0) && ! ( dfj <= 0 ) ) 
// 	std::cout << " nan : dfj = " << dfj << " fval =  " << fval << " invF = " << invFval << " grad = " << mfg[j] << " par[j] = " << par[j] << std::endl;  

      //std::cout << " x = "  << currentPosition[0] <<  " par[j] = " << par[j] << " : dfj = " << dfj << " fval =  " << fval << " invF = " << invFval << " grad = " << mfg[j] << " deriv = " << grad[j] << std::endl;  

      
      // in second derivative use Fumili approximation neglecting the term containing the 
      // second derivatives of the model function
      for (int k = j; k < npar; ++ k) { 
	int idx =  j + k*(k+1)/2; 
	if (fabs( mfg[k]) < minDouble ) { 
	  if (mfg[k] < 0)  
	    mfg[k] =  -minDouble;
	  else 
	    mfg[k] =  minDouble;
	}
	  
	double dfk =  invFval * mfg[k]; 
	// avoid that dfk*dfj are one small and one huge so I get a nan 
      // to avoid summing infinite and nan later when calculating the hessian
	if ( fabs(dfk) > maxDouble2 ) { 
	  if (dfk > 0) 
	    dfk = maxDouble2; 
	  else 
	    dfk = -maxDouble2;
	}

	
	h[idx] += dfj * dfk; 
// 	if ( ( ! ( h[idx] > 0) && ! ( h[idx] <= 0 ) ) ) 
// 	  std::cout << " nan : dfj = " << dfj << " fval =  " << fval << " invF = " << invFval << " gradj = " << mfg[j] 
// 		    << " dfk = " << dfk << " gradk =  "<< mfg[k]  << " hess_jk = " << h[idx] << " par[k] = " << par[k] << std::endl;    
      }
      
    } // end param loop  
  
  } // end points loop 

//   std::cout <<"\nEVALUATED GRADIENT and HESSIAN " << std::endl;
//   for (int j = 0; j < npar; ++j) { 
//     std::cout << " j = " << j << " grad = " << grad[j] << std::endl; 
//     for (int k = j; k < npar; ++k) { 
//       std::cout << " k = " << k << " hess = " << hessian(j,k) << "  " << h[ j + k*(k+1)/2]  << std::endl; 
//     }
//   }

    // set value in base class
  setFCNValue( logLikelihood); 

}
