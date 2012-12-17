#ifndef MA_LaOuterProd_H_
#define MA_LaOuterProd_H_

/** LAPACK Algebra
    specialize the outer_product function for LAVector;
 */

#include "Minuit/VectorOuterProduct.h"
#include "Minuit/ABSum.h"
#include "Minuit/LAVector.h"
#include "Minuit/LASymMatrix.h"

inline ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, double>, double>, double> outer_product(const ABObj<vec, LAVector, double>& obj) {
//   std::cout<<"ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, double>, double>, double> outer_product(const ABObj<vec, LAVector, double>& obj)"<<std::endl;
  return ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, double>, double>, double>(VectorOuterProduct<ABObj<vec, LAVector, double>, double>(obj));
}

// f*outer
template<class T>
inline ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T> operator*(T f, const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T>& obj) {
//   std::cout<<"ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T> operator*(T f, const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T>& obj)"<<std::endl;
  return ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T>(obj.obj(), obj.f()*f);
}

// outer/f
template<class T>
inline ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T> operator/(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T>& obj, T f) {
//   std::cout<<"ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T> operator/(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T>& obj, T f)"<<std::endl;
  return ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T>(obj.obj(), obj.f()/f);
}
 
// -outer
template<class T>
inline ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T> operator-(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T>& obj) {
//   std::cout<<"ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T> operator/(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T>& obj, T f)"<<std::endl;
  return ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T>(obj.obj(), T(-1.)*obj.f());
}

void outer_prod(LASymMatrix&, const LAVector&, double f = 1.);

#endif //MA_LaOuterProd_H_
