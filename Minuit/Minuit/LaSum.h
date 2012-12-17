#ifndef LA_LaSum_H_
#define LA_LaSum_H_

#include "Minuit/ABSum.h"
#include "Minuit/LAVector.h"
#include "Minuit/LASymMatrix.h"

#define OP_ADD1(MT,MAT1,T) \
inline ABObj<MT,MAT1,T>  operator-(const ABObj<MT,MAT1,T>& m) {\
  return ABObj<MT,MAT1,T> (m.obj(), T(-1.)*m.f());\
}\
			  \
inline ABObj<MT,ABSum<ABObj<MT,MAT1,T>, ABObj<MT,MAT1,T> >,T>  operator+(const ABObj<MT,MAT1,T>& a, const ABObj<MT,MAT1,T>& b) {	  \
  return ABObj<MT,ABSum<ABObj<MT,MAT1,T>, ABObj<MT,MAT1,T> >,T>(ABSum<ABObj<MT,MAT1,T>, ABObj<MT,MAT1,T> >(a, b));			  \
}													       \
inline ABObj<MT,ABSum<ABObj<MT,MAT1,T>, ABObj<MT,MAT1,T> >,T>  operator-(const ABObj<MT,MAT1,T>& a, const ABObj<MT,MAT1,T>& b) {	       \
  return ABObj<MT,ABSum<ABObj<MT,MAT1,T>, ABObj<MT,MAT1,T> >,T>(ABSum<ABObj<MT,MAT1,T>, ABObj<MT,MAT1,T> >(a,ABObj<MT,MAT1,T> (b.obj(),T(-1.)*b.f())));	       \
}

OP_ADD1(vec,LAVector,double)
OP_ADD1(sym,LASymMatrix,double)

#define OP_SCALE(MT,MAT1,T) \
inline ABObj<MT,MAT1,T> operator*(T f, const MAT1& obj) { \
  return ABObj<MT,MAT1,T>(obj, f); \
}

OP_SCALE(sym,LASymMatrix,double)
OP_SCALE(vec,LAVector,double)

#define OP_SCALE1(MT,MAT1,T) \
inline ABObj<MT,MAT1,T> operator/(const MAT1& obj, T f) { \
  return ABObj<MT,MAT1,T>(obj, 1./f); \
}

OP_SCALE1(sym,LASymMatrix,double)
OP_SCALE1(vec,LAVector,double)

#define OP_MIN(MT,MAT1,T) \
inline ABObj<MT,MAT1,T> operator-(const MAT1& obj) { \
  return ABObj<MT,MAT1,T>(obj, T(-1.)); \
}

OP_MIN(sym,LASymMatrix,double)
OP_MIN(vec,LAVector,double)

#endif //LA_LaSum_H_
