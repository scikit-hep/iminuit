#ifndef MN_LaInverse_H_
#define MN_LaInverse_H_

/** LAPACK Algebra
    specialize the invert function for LASymMatrix 
 */

#include "Minuit/MatrixInverse.h"
#include "Minuit/LASymMatrix.h"

inline ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, double>, double>, double> inverse(const ABObj<sym, LASymMatrix, double>& obj) {
  return ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, double>, double>, double>(MatrixInverse<sym, ABObj<sym, LASymMatrix, double>, double>(obj));
}

template<class T>
inline ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, double>, double>, double> operator*(T f, const  ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, double>, double>, double>& inv) {
  return ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, double>, double>, double>(inv.obj(), f*inv.f());
}

template<class T>
inline ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, double>, double>, double> operator/(const  ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, double>, double>, double>& inv, T f) {
  return ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, double>, double>, double>(inv.obj(), inv.f()/f);
}

template<class T>
inline ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, double>, double>, double> operator-(const  ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, double>, double>, double>& inv) {
  return ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, double>, double>, double>(inv.obj(), T(-1.)*inv.f());
}

int invert(LASymMatrix&);

int invert_undef_sym(LASymMatrix&);

/*
template<class M>
inline ABObj<sym, MatrixInverse<sym, ABObj<sym, M, double>, double>, double> inverse(const ABObj<sym, M, double>& obj) {
  return ABObj<sym, MatrixInverse<sym, ABObj<sym, M, double>, double>, double>(MatrixInverse<sym, ABObj<sym, M, double>, double>(obj));
}

inline ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, double>, double>, double> inverse(const ABObj<sym, LASymMatrix, double>& obj) {
  return ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, double>, double>, double>(MatrixInverse<sym, ABObj<sym, LASymMatrix, double>, double>(obj));
}
*/

#endif //MN_LaInverse_H_
