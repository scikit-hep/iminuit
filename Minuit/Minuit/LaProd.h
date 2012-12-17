#ifndef LA_LaProd_H_
#define LA_LaProd_H_

#include "Minuit/ABProd.h"
#include "Minuit/LAVector.h"
#include "Minuit/LASymMatrix.h"

/*
  LM" remove this for fixing alpha poblem
#define OP_MULT1(MT1,MT2,MAT1,MAT2,T) \
template<class B>                     \
inline ABObj<AlgebraicProdType<MT1,MT2>::Type,ABProd<ABObj<MT1,MAT1,T>, ABObj<MT2,B,T> >,T>  operator*(const ABObj<MT1,MAT1,T>& a, const ABObj<MT2,B,T>& b) { return ABObj<AlgebraicProdType<MT1,MT2>::Type,ABProd<ABObj<MT1,MAT1,T>, ABObj<MT2,B,T> >,T>(ABProd<ABObj<MT1,MAT1,T>, ABObj<MT2,B,T> >(a, b)); }			\
template<class A> 										  \
inline ABObj<AlgebraicProdType<MT1,MT2>::Type,ABProd<ABObj<MT1,A,T>, ABObj<MT2,MAT2,T> >,T>  operator*(const ABObj<MT1,A,T>& a, const ABObj<MT2,MAT2,T>& b) { \
  return ABObj<AlgebraicProdType<MT1,MT2>::Type,ABProd<ABObj<MT1,A,T>, ABObj<MT2,MAT2,T> >,T>(ABProd<ABObj<MT1,A,T>, ABObj<MT2,MAT2,T> >(a, b));		  \
} \
  \
*/

#define OP_MULT1(MT1,MT2,MAT1,MAT2,T) \
inline ABObj<AlgebraicProdType<MT1,MT2>::Type,ABProd<ABObj<MT1,MAT1,T>, ABObj<MT2,MAT2,T> >,T>  operator*(const ABObj<MT1,MAT1,T>& a, const ABObj<MT2,MAT2,T>& b) {	  \
  return ABObj<AlgebraicProdType<MT1,MT2>::Type,ABProd<ABObj<MT1,MAT1,T>, ABObj<MT2,MAT2,T> >,T>(ABProd<ABObj<MT1,MAT1,T>, ABObj<MT2,MAT2,T> >(a, b));	\
}			

OP_MULT1(sym,vec,LASymMatrix,LAVector,double)
// OP_MULT1(sym,gen,LASymMatrix,LAGenMatrix,double)
// OP_MULT1(sym,sym,LASymMatrix,LASymMatrix,double)
// OP_MULT1(gen,vec,LAGenMatrix,LAVector,double)
// OP_MULT1(gen,sym,LAGenMatrix,LASymMatrix,double)
// OP_MULT1(gen,gen,LAGenMatrix,LAGenMatrix,double)

#endif //LA_LaProd_H_
