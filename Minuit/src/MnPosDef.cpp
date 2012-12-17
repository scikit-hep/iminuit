#include "Minuit/MnPosDef.h"
#include "Minuit/MinimumState.h"
#include "Minuit/MnMachinePrecision.h"
#include "Minuit/MnPrint.h"

//#include "Minuit/MnPrint.h"

#include <algorithm>


LAVector eigenvalues(const LASymMatrix&);


MinimumState MnPosDef::operator()(const MinimumState& st, const MnMachinePrecision& prec) const {
  
  MinimumError err = (*this)(st.error(), prec);
  return MinimumState(st.parameters(), err, st.gradient(), st.edm(), st.nfcn());
}

MinimumError MnPosDef::operator()(const MinimumError& e, const MnMachinePrecision& prec) const {

  MnAlgebraicSymMatrix err(e.invHessian());
  if(err.size() == 1 && err(0,0) < prec.eps()) {
    err(0,0) = 1.;
    return MinimumError(err, MinimumError::MnMadePosDef());
  } 
  if(err.size() == 1 && err(0,0) > prec.eps()) {
    return e;
  } 
//   std::cout<<"MnPosDef init matrix= "<<err<<std::endl;

  double epspdf = std::max(1.e-6, prec.eps2());
  double dgmin = err(0,0);

  for(unsigned int i = 0; i < err.nrow(); i++) {
    if(err(i,i) < prec.eps2()) std::cout<<"negative or zero diagonal element "<<i<<" in covariance matrix"<<std::endl;
    if(err(i,i) < dgmin) dgmin = err(i,i);
  }
  double dg = 0.;
  if(dgmin < prec.eps2()) {
    //dg = 1. + epspdf - dgmin; 
    dg = 0.5 + epspdf - dgmin; 
//     dg = 0.5*(1. + epspdf - dgmin); 
    std::cout<<"added "<<dg<<" to diagonal of error matrix"<<std::endl;
    //std::cout << "Error matrix " << err << std::endl;
  }

  MnAlgebraicVector s(err.nrow());
  MnAlgebraicSymMatrix p(err.nrow());
  for(unsigned int i = 0; i < err.nrow(); i++) {
    err(i,i) += dg;
    if(err(i,i) < 0.) err(i,i) = 1.;
    s(i) = 1./sqrt(err(i,i));
    for(unsigned int j = 0; j <= i; j++) {
      p(i,j) = err(i,j)*s(i)*s(j);
    }
  }
  
  //std::cout<<"MnPosDef p: "<<p<<std::endl;
  MnAlgebraicVector eval = eigenvalues(p);
  double pmin = eval(0);
  double pmax = eval(eval.size() - 1);
  //std::cout<<"pmin= "<<pmin<<" pmax= "<<pmax<<std::endl;
  pmax = std::max(fabs(pmax), 1.);
  if(pmin > epspdf*pmax) return MinimumError(err, e.dcovar());
  
  double padd = 0.001*pmax - pmin;
  std::cout<<"eigenvalues: "<<std::endl;
  for(unsigned int i = 0; i < err.nrow(); i++) {
    err(i,i) *= (1. + padd);
    std::cout<<eval(i)<<std::endl;
  }
//   std::cout<<"MnPosDef final matrix: "<<err<<std::endl;
  std::cout<<"matrix forced pos-def by adding "<<padd<<" to diagonal"<<std::endl;
//   std::cout<<"eigenvalues: "<<eval<<std::endl;
  return MinimumError(err, MinimumError::MnMadePosDef());
}
