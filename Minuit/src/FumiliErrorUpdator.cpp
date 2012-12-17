#include "Minuit/FumiliErrorUpdator.h"
#include "Minuit/MnFcn.h"
#include "Minuit/MnStrategy.h"
#include "Minuit/MnUserParameterState.h"
#include "Minuit/FumiliGradientCalculator.h"
#include "Minuit/MinimumParameters.h"
#include "Minuit/FunctionGradient.h"
#include "Minuit/MnMatrix.h"
#include "Minuit/MinimumError.h"
#include "Minuit/MinimumState.h"
#include "Minuit/LaSum.h"
#include "Minuit/MnPrint.h"


double sum_of_elements(const LASymMatrix&);


MinimumError FumiliErrorUpdator::update(const MinimumState& s0, 
					 const MinimumParameters& p1,
					 const FunctionGradient& g1) const {

  // dummy methods to suppress unused variable warnings
  // this member function should never be called within
  // the Fumili method...
  s0.fval();
  p1.fval();
  g1.isValid();
  return MinimumError(2);
}


MinimumError FumiliErrorUpdator::update(const MinimumState& s0, 
					 const MinimumParameters& p1,
					const GradientCalculator&  gc , 
					double lambda) const {


  // need to downcast to FumiliGradientCalculator
  FumiliGradientCalculator * fgc = dynamic_cast< FumiliGradientCalculator *>( const_cast<GradientCalculator *>(&gc) ); 
  assert(fgc != 0); 
  

  // get hessian from gradient calculator

  MnAlgebraicSymMatrix h = fgc->hessian(); 

  int nvar = p1.vec().size();

  // apply Marquard lambda factor 
  for (int j = 0; j < nvar; j++) { 
    h(j,j) *= (1. + lambda);
    // if h(j,j) is zero what to do ?
    if ( fabs( h(j,j) ) < 1E-300 ) { // should use DBL_MIN 
      // put a cut off to avoid zero on diagonals
      if ( lambda > 1) 
	h(j,j) = lambda*1E-300; 
      else 
	h(j,j) = 1E-300; 
    }
  }
 

 
  int ifail = invert(h);
  if(ifail != 0) {
    std::cout<<"FumiliErrorUpdator inversion fails; return diagonal matrix."<<std::endl;
    for(unsigned int i = 0; i < h.nrow(); i++) {
      h(i,i) = 1./h(i,i);
    }
  }


  const MnAlgebraicSymMatrix& V0 = s0.error().invHessian();

  // calculate by how much did the covariance matrix changed
  // (if it changed a lot since the last step, probably we 
  // are not yet near the minimum)
  double dcov = 0.5*(s0.error().dcovar() + sum_of_elements(h-V0)/sum_of_elements(h)); 



  return MinimumError(h, dcov);

}

