#ifndef _AB_ABSum_H_
#define _AB_ABSum_H_

#include "Minuit/ABObj.h"

template<class M1, class M2>
class ABSum {

private:

  ABSum() : theA(M1()), theB(M2()) {}

  ABSum& operator=(const ABSum&) {return *this;}

  template<class A, class B>
  ABSum& operator=(const ABSum<A,B>&) {return *this;}

public:

  ABSum(const M1& a, const M2& b): theA(a), theB(b) {}

  ~ABSum() {}

  ABSum(const ABSum& sum) : theA(sum.theA), theB(sum.theB) {}

  template<class A, class B>
  ABSum(const ABSum<A,B>& sum) : theA(M1(sum.theA)), theB(M2(sum.theB)) {}

  const M1& a() const {return theA;}
  const M2& b() const {return theB;}

private:

  M1 theA;
  M2 theB;
};

// ABObj + ABObj
template<class atype, class A, class btype, class B, class T>
inline ABObj<typename AlgebraicSumType<atype, btype>::Type, ABSum<ABObj<atype,A,T>, ABObj<btype,B,T> >,T> operator+(const ABObj<atype,A,T>& a, const ABObj<btype,B,T>& b) {

  return ABObj<typename AlgebraicSumType<atype,btype>::Type, ABSum<ABObj<atype,A,T>, ABObj<btype,B,T> >,T>(ABSum<ABObj<atype,A,T>, ABObj<btype,B,T> >(a, b));
}

// ABObj - ABObj
template<class atype, class A, class btype, class B, class T>
inline ABObj<typename AlgebraicSumType<atype, btype>::Type, ABSum<ABObj<atype,A,T>, ABObj<btype,B,T> >,T> operator-(const ABObj<atype,A,T>& a, const ABObj<btype,B,T>& b) {

  return ABObj<typename AlgebraicSumType<atype,btype>::Type, ABSum<ABObj<atype,A,T>, ABObj<btype,B,T> >,T>(ABSum<ABObj<atype,A,T>, ABObj<btype,B,T> >(a, ABObj<btype,B,T>(b.obj(), T(-1.)*b.f())));
}

#endif //_AB_ABSum_H_
