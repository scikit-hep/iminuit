#include "Minuit/FumiliGradientCalculator.h"
#include "Minuit/FumiliFCNBase.h"
#include "Minuit/MnUserTransformation.h"
#include "Minuit/FunctionGradient.h"
#include "Minuit/MinimumParameters.h"
#include "Minuit/FumiliChi2FCN.h"
#include "Minuit/FumiliMaximumLikelihoodFCN.h"

//to compare with N2P calculator
//#define DEBUG 1
#ifdef DEBUG
#include "Minuit/MnPrint.h"
#include "Minuit/Numerical2PGradientCalculator.h"
#include "Minuit/MnStrategy.h"
#include "Minuit/MnUserFcn.h"
#endif

FunctionGradient FumiliGradientCalculator::operator()(const MinimumParameters& par) const {

  int nvar = par.vec().size();
  std::vector<double> extParam = theTransformation(  par.vec() );
//   std::vector<double> deriv; 
//   deriv.reserve( nvar ); 
//   for (int i = 0; i < nvar; ++i) {
//     unsigned int ext = theTransformation.extOfInt(i);
//     if ( theTransformation.parameter(ext).hasLimits()) 
//       deriv.push_back( theTransformation.dInt2Ext( i, par.vec()(i) ) );
//     else 
//       deriv.push_back(1.0); 
//   }

  // eval gradient 
  FumiliFCNBase & fcn = const_cast<FumiliFCNBase &>(theFcn);  

  fcn.evaluateAll(extParam);


  MnAlgebraicVector v(nvar);
  MnAlgebraicSymMatrix h(nvar);


  const std::vector<double> & fcn_gradient = theFcn.gradient(); 
  assert( fcn_gradient.size() == extParam.size() ); 


//   for (int i = 0; i < nvar; ++i) { 
//     unsigned int iext = theTransformation.extOfInt(i);    
//     double ideriv = 1.0; 
//     if ( theTransformation.parameter(iext).hasLimits()) 
//       ideriv =  theTransformation.dInt2Ext( i, par.vec()(i) ) ;


//     //     v(i) = fcn_gradient[iext]*deriv;
//     v(i) = ideriv*fcn_gradient[iext];

//     for (int j = i; j < nvar; ++j) { 
//       unsigned int jext = theTransformation.extOfInt(j);
//       double jderiv = 1.0; 
//       if ( theTransformation.parameter(jext).hasLimits()) 
// 	jderiv =  theTransformation.dInt2Ext( j, par.vec()(j) ) ;
      
// //       h(i,j) = deriv[i]*deriv[j]*theFcn.hessian(iext,jext); 
//       h(i,j) = ideriv*jderiv*theFcn.hessian(iext,jext); 
//     }
//   }


  // cache deriv and index values . 
  // in large parameter limit then need to re-optimize and see if better not caching

  std::vector<double> deriv(nvar); 
  std::vector<unsigned int> extIndex(nvar); 
  for (int i = 0; i < nvar; ++i) { 
    extIndex[i] = theTransformation.extOfInt(i);    
    deriv[i] = 1;
    if ( theTransformation.parameter(extIndex[i]).hasLimits()) 
      deriv[i] =  theTransformation.dInt2Ext( i, par.vec()(i) ) ;

    v(i) = fcn_gradient[extIndex[i]]*deriv[i];

    for (int j = 0; j <= i; ++j) {       
       h(i,j) = deriv[i]*deriv[j]*theFcn.hessian(extIndex[i],extIndex[j]); 
    }
  }

#ifdef DEBUG
  // compare with other gradient 
//   // calculate gradient using Minuit method 
  
  Numerical2PGradientCalculator gc(MnUserFcn(theFcn,theTransformation), theTransformation, MnStrategy(1));
  FunctionGradient g2 = gc(par);

  std::cout << "Fumili gradient " << v << std::endl;
  std::cout << "Minuit gradient " << g2.vec() << std::endl;
#endif  

  // store calculated hessian
  theHessian = h; 
  return FunctionGradient(v); 
}

FunctionGradient FumiliGradientCalculator::operator()(const MinimumParameters& par,
				      const FunctionGradient&) const

{ 
  return this->operator()(par); 

}
