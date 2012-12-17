#include "Minuit/NegativeG2LineSearch.h"
#include "Minuit/MnFcn.h"
#include "Minuit/MinimumState.h"
#include "Minuit/GradientCalculator.h"
#include "Minuit/MnMachinePrecision.h"
#include "Minuit/MnLineSearch.h"
#include "Minuit/MnParabolaPoint.h"
#include "Minuit/VariableMetricEDMEstimator.h"

/**
   when the second derivatives are negative perform a  line search  along parameter which gives negative second derivative 
  and magnitude  equal to the gradient step size. Recalculate the gradients for all the parameter after the correction and 
  continue iteration in case the second derivatives are still negative
*/


MinimumState NegativeG2LineSearch::operator()(const MnFcn& fcn, const MinimumState& st, const  GradientCalculator& gc, const MnMachinePrecision& prec) const {

  bool negG2 = hasNegativeG2(st.gradient(), prec);
  if(!negG2) return st;

  unsigned int n = st.parameters().vec().size();
  FunctionGradient dgrad = st.gradient();
  MinimumParameters pa = st.parameters();
  bool iterate = false;
  unsigned int iter = 0;
  do {
    iterate = false;
    for(unsigned int i = 0; i < n; i++) {

//       std::cout << "negative G2 - iter " << iter << " param " << i << "  grad2 " << dgrad.g2()(i) << " grad " << dgrad.vec()(i) 
// 		<< " grad step " << dgrad.gstep()(i) << std::endl; 
      if(dgrad.g2()(i) <= 0) {      
//       if(dgrad.g2()(i) < prec.eps()) {
	// do line search if second derivative negative
	MnAlgebraicVector step(n);
	MnLineSearch lsearch;
 	step(i) = dgrad.gstep()(i)*dgrad.vec()(i);
	//	if(fabs(dgrad.vec()(i)) >  prec.eps2()) 
	if(fabs(dgrad.vec()(i)) >  0 ) 
	  step(i) *= (-1./fabs(dgrad.vec()(i)));
	double gdel = step(i)*dgrad.vec()(i);
	MnParabolaPoint pp = lsearch(fcn, pa, step, gdel, prec);
	//	std::cout << " line search result " << pp.x() << "  " << pp.y() << std::endl;
	step *= pp.x();
	pa = MinimumParameters(pa.vec() + step, pp.y());    
	dgrad = gc(pa, dgrad);         
//  	std::cout << "Line search - iter" << iter << " param " << i << " step " << step(i) << " new grad2 " << dgrad.g2()(i) << " new grad " <<  dgrad.vec()(i) << std::endl;
	iterate = true;
	break;
      } 
    }
  } while(iter++ < 2*n && iterate);

  MnAlgebraicSymMatrix mat(n);
  for(unsigned int i = 0; i < n; i++)	
    mat(i,i) = (fabs(dgrad.g2()(i)) > prec.eps2() ? 1./dgrad.g2()(i) : 1.);

  MinimumError err(mat, 1.);
  double edm = VariableMetricEDMEstimator().estimate(dgrad, err);

  return MinimumState(pa, err, dgrad, edm, fcn.numOfCalls());
}

bool NegativeG2LineSearch::hasNegativeG2(const FunctionGradient& grad, const MnMachinePrecision& /*prec */ ) const {
  
  for(unsigned int i = 0; i < grad.vec().size(); i++) 
    //     if(grad.g2()(i) < prec.eps2()) { 
    if(grad.g2()(i) <= 0 ) { 
      //      std::cout << "negative G2 " << i << "  grad " << grad.g2()(i) << " precision " << prec.eps2() << std::endl;
      return true;
    }

  return false;
}
