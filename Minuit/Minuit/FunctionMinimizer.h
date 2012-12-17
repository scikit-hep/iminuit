#ifndef MN_FunctionMinimizer_H_
#define MN_FunctionMinimizer_H_


#include "Minuit/MnConfig.h"
#include <vector>

class FCNBase;
class FCNGradientBase;
class FunctionMinimum;


/** base class for function minimizers; user may give FCN or FCN with gradient,
    parameter starting values and initial error guess (sigma) (or "step size"),
    or parameter starting values and initial covariance matrix; 
    covariance matrix is stored in upper triangular packed storage format, 
    e.g. the elements in the array are arranged like 
    {a(0,0), a(0,1), a(1,1), a(0,2), a(1,2), a(2,2), ...},
    the size is nrow*(nrow+1)/2 (see also MnUserCovariance.h);   
 */

class FunctionMinimizer {

public:
  
  virtual ~FunctionMinimizer() {}

  //starting values for parameters and errors
  virtual FunctionMinimum minimize(const FCNBase&, const std::vector<double>& par, const std::vector<double>& err, unsigned int strategy, unsigned int maxfcn, double toler) const = 0; 

  //starting values for parameters and errors and FCN with gradient
  virtual FunctionMinimum minimize(const FCNGradientBase&, const std::vector<double>& par, const std::vector<double>& err, unsigned int strategy, unsigned int maxfcn, double toler) const = 0; 

  //starting values for parameters and covariance matrix
  virtual FunctionMinimum minimize(const FCNBase&, const std::vector<double>& par, unsigned int nrow, const std::vector<double>& cov, unsigned int strategy, unsigned int maxfcn, double toler) const = 0; 

  //starting values for parameters and covariance matrix and FCN with gradient
  virtual FunctionMinimum minimize(const FCNGradientBase&, const std::vector<double>& par, unsigned int nrow, const std::vector<double>& cov, unsigned int strategy, unsigned int maxfcn, double toler) const = 0; 

};

#endif //MN_FunctionMinimizer_H_
