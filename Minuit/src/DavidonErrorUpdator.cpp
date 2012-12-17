#include "Minuit/DavidonErrorUpdator.h"
#include "Minuit/MinimumState.h"
#include "Minuit/LaSum.h"
#include "Minuit/LaProd.h"

double inner_product(const LAVector&, const LAVector&);
double similarity(const LAVector&, const LASymMatrix&);
double sum_of_elements(const LASymMatrix&);

MinimumError DavidonErrorUpdator::update(const MinimumState& s0, 
					 const MinimumParameters& p1,
					 const FunctionGradient& g1) const {

  const MnAlgebraicSymMatrix& V0 = s0.error().invHessian();
  MnAlgebraicVector dx = p1.vec() - s0.vec();
  MnAlgebraicVector dg = g1.vec() - s0.gradient().vec();
  
  double delgam = inner_product(dx, dg);
  double gvg = similarity(dg, V0);

//   std::cout<<"delgam= "<<delgam<<" gvg= "<<gvg<<std::endl;
  MnAlgebraicVector vg = V0*dg;

  MnAlgebraicSymMatrix Vupd = outer_product(dx)/delgam - outer_product(vg)/gvg;

  if(delgam > gvg) {
    Vupd += gvg*outer_product(MnAlgebraicVector(dx/delgam - vg/gvg));
  }

  double sum_upd = sum_of_elements(Vupd);
  Vupd += V0;
  
  double dcov = 0.5*(s0.error().dcovar() + sum_upd/sum_of_elements(Vupd));
  
  return MinimumError(Vupd, dcov);
}

/*
MinimumError DavidonErrorUpdator::update(const MinimumState& s0, 
					 const MinimumParameters& p1,
					 const FunctionGradient& g1) const {

  const MnAlgebraicSymMatrix& V0 = s0.error().invHessian();
  MnAlgebraicVector dx = p1.vec() - s0.vec();
  MnAlgebraicVector dg = g1.vec() - s0.gradient().vec();
  
  double delgam = inner_product(dx, dg);
  double gvg = similarity(dg, V0);

//   std::cout<<"delgam= "<<delgam<<" gvg= "<<gvg<<std::endl;
  MnAlgebraicVector vg = V0*dg;
//   MnAlgebraicSymMatrix Vupd(V0.nrow());

//   MnAlgebraicSymMatrix dd = ( 1./delgam )*outer_product(dx);
//   dd *= ( 1./delgam );
//   MnAlgebraicSymMatrix VggV = ( 1./gvg )*outer_product(vg);
//   VggV *= ( 1./gvg );
//   Vupd = dd - VggV;
//   MnAlgebraicSymMatrix Vupd = ( 1./delgam )*outer_product(dx) - ( 1./gvg )*outer_product(vg);
  MnAlgebraicSymMatrix Vupd = outer_product(dx)/delgam - outer_product(vg)/gvg;
  
  if(delgam > gvg) {
//     dx *= ( 1./delgam );
//     vg *= ( 1./gvg );
//     MnAlgebraicVector flnu = dx - vg;
//     MnAlgebraicSymMatrix tmp = outer_product(flnu);
//     tmp *= gvg;
//     Vupd = Vupd + tmp;
    Vupd += gvg*outer_product(dx/delgam - vg/gvg);
  }

//   
//     MnAlgebraicSymMatrix dd = outer_product(dx);
//     dd *= ( 1./delgam );
//     MnAlgebraicSymMatrix VggV = outer_product(vg);
//     VggV *= ( 1./gvg );
//     Vupd = dd - VggV;
//   
//     
//   double phi = delgam/(delgam - gvg);

//   MnAlgebraicSymMatrix Vupd(V0.nrow());
//   if(phi < 0) {
//     // rank-2 update
//     MnAlgebraicSymMatrix dd = outer_product(dx);
//     dd *= ( 1./delgam );
//     MnAlgebraicSymMatrix VggV = outer_product(vg);
//     VggV *= ( 1./gvg );
//     Vupd = dd - VggV;
//   }
//   if(phi > 1) {
//     // rank-1 update
//     MnAlgebraicVector tmp = dx - vg;
//     Vupd = outer_product(tmp);
//     Vupd *= ( 1./(delgam - gvg) );
//   }
//     

//     
//   if(delgam > gvg) {
//     // rank-1 update
//     MnAlgebraicVector tmp = dx - vg;
//     Vupd = outer_product(tmp);
//     Vupd *= ( 1./(delgam - gvg) );
//   } else { 
//     // rank-2 update
//     MnAlgebraicSymMatrix dd = outer_product(dx);
//     dd *= ( 1./delgam );
//     MnAlgebraicSymMatrix VggV = outer_product(vg);
//     VggV *= ( 1./gvg );
//     Vupd = dd - VggV;
//   }
//   

  double sum_upd = sum_of_elements(Vupd);
  Vupd += V0;
    
//   MnAlgebraicSymMatrix V1 = V0 + Vupd;

  double dcov = 
    0.5*(s0.error().dcovar() + sum_upd/sum_of_elements(Vupd));
  
  return MinimumError(Vupd, dcov);
}
*/
