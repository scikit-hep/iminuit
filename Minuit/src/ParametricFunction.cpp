#include "Minuit/ParametricFunction.h"
#include "Minuit/MnFcn.h"
#include "Minuit/MnStrategy.h"
#include "Minuit/MnUserParameterState.h"
#include "Minuit/Numerical2PGradientCalculator.h"
#include "Minuit/FunctionGradient.h"
#include "Minuit/MnVectorTransform.h"
//#include "Minuit/MnPrint.h"



std::vector<double>  ParametricFunction::getGradient(const std::vector<double>& x) const { 


  //LM:  this I believe is not very efficient
  MnFcn mfcn(*this);
  
  MnStrategy strategy(1);

  // ????????? how does it know the transformation????????
  std::vector<double> err(x.size());
  err.assign(x.size(), 0.1);
  // need to use parameters 
  MnUserParameterState st(x, err);
  
  Numerical2PGradientCalculator gc(mfcn, st.trafo(), strategy);
  FunctionGradient g = gc(x); 
  const MnAlgebraicVector & grad = g.vec();
  assert( grad.size() == x.size() );
  MnVectorTransform vt; 
  //  std::cout << "Param Function gradient " << grad << std::endl; 
  return vt( grad ); 
}
