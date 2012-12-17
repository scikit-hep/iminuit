#ifndef _AB_ABProd_H_
#define _AB_ABProd_H_

#include "Minuit/ABObj.h"

template<class M1, class M2>
class ABProd {

private:

  ABProd() : theA(M1()), theB(M2()) {}

  ABProd& operator=(const ABProd&) {return *this;}

  template<class A, class B>
  ABProd& operator=(const ABProd<A,B>&) {return *this;}
  
public:

  ABProd(const M1& a, const M2& b): theA(a), theB(b) {}

  ~ABProd() {}

  ABProd(const ABProd& prod) : theA(prod.theA), theB(prod.theB) {}

  template<class A, class B>
  ABProd(const ABProd<A,B>& prod) : theA(M1(prod.theA)), theB(M2(prod.theB)) {}

  const M1& a() const {return theA;}
  const M2& b() const {return theB;}
 
private:

  M1 theA;
  M2 theB;
};

// ABObj * ABObj
template<class atype, class A, class btype, class B, class T>
inline ABObj<typename AlgebraicProdType<atype, btype>::Type, ABProd<ABObj<atype,A,T>, ABObj<btype,B,T> >,T> operator*(const ABObj<atype,A,T>& a, const ABObj<btype,B,T>& b) {

  return ABObj<typename AlgebraicProdType<atype,btype>::Type, ABProd<ABObj<atype,A,T>, ABObj<btype,B,T> >,T>(ABProd<ABObj<atype,A,T>, ABObj<btype,B,T> >(a, b));
}

#endif //_AB_ABProd_H_
