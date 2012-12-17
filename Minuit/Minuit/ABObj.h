#ifndef AB_ABObj_H_
#define AB_ABObj_H_

#include "Minuit/ABTypes.h"

template<class mtype, class M, class T>
class ABObj {

public:

  typedef mtype Type;

private:

  ABObj() : theObject(M()), theFactor(T(0.)) {}

  ABObj& operator=(const ABObj&) {return *this;}

  template<class a, class b, class c>
  ABObj(const ABObj<a,b,c>& obj) : theObject(M()), theFactor(T(0.)) {}

  template<class a, class b, class c>
  ABObj& operator=(const ABObj<a,b,c>&) {return *this;}
  
public:

  ABObj(const M& obj) : theObject(obj), theFactor(T(1.)) {}

  ABObj(const M& obj, T factor) : theObject(obj), theFactor(factor) {}

  ~ABObj() {}

  ABObj(const ABObj& obj) : 
    theObject(obj.theObject), theFactor(obj.theFactor) {}

  template<class b, class c>
  ABObj(const ABObj<mtype,b,c>& obj) : 
    theObject(M(obj.theObject)), theFactor(T(obj.theFactor)) {}

  const M& obj() const {return theObject;}

  T f() const {return theFactor;}

private:

  M theObject;
  T theFactor;
};

class LAVector;
template <> class ABObj<vec, LAVector, double> {

public:

  typedef vec Type;

private:

  ABObj& operator=(const ABObj&) {return *this;}
  
public:

  ABObj(const LAVector& obj) : theObject(obj), theFactor(double(1.)) {}

  ABObj(const LAVector& obj, double factor) : theObject(obj), theFactor(factor) {}

  ~ABObj() {}

  // remove copy constructure to fix a problem in AIX 
  // should be able to use the compiler generated one
//   ABObj(const ABObj& obj) : 
//     theObject(obj.theObject), theFactor(obj.theFactor) {}

  template<class c>
  ABObj(const ABObj<vec,LAVector,c>& obj) : 
    theObject(obj.theObject), theFactor(double(obj.theFactor)) {}

  const LAVector& obj() const {return theObject;}

  double f() const {return theFactor;}

private:

  const LAVector& theObject;
  double theFactor;
};

class LASymMatrix;
template <> class ABObj<sym, LASymMatrix, double> {

public:

  typedef sym Type;

private:

  ABObj& operator=(const ABObj&) {return *this;}
  
public:

  ABObj(const LASymMatrix& obj) : theObject(obj), theFactor(double(1.)) {}

  ABObj(const LASymMatrix& obj, double factor) : theObject(obj), theFactor(factor) {}

  ~ABObj() {}

  ABObj(const ABObj& obj) : 
    theObject(obj.theObject), theFactor(obj.theFactor) {}

  template<class c>
  ABObj(const ABObj<vec,LASymMatrix,c>& obj) : 
    theObject(obj.theObject), theFactor(double(obj.theFactor)) {}

  const LASymMatrix& obj() const {return theObject;}

  double f() const {return theFactor;}

private:

  const LASymMatrix& theObject;
  double theFactor;
};

// templated scaling operator *
template<class mt, class M, class T>
inline ABObj<mt, M, T> operator*(T f, const M& obj) {
  return ABObj<mt, M, T>(obj, f);
}

// templated operator /
template<class mt, class M, class T>
inline ABObj<mt, M, T> operator/(const M& obj, T f) {
  return ABObj<mt, M, T>(obj, T(1.)/f);
}

// templated unary operator -
template<class mt, class M, class T>
inline ABObj<mt,M,T> operator-(const M& obj) {
  return ABObj<mt,M,T>(obj, T(-1.));
}

/*
// specialization for LAVector

inline ABObj<vec, LAVector, double> operator*(double f, const LAVector& obj) {
  return ABObj<vec, LAVector, double>(obj, f);
}

inline ABObj<vec, LAVector, double> operator/(const LAVector& obj, double f) {
  return ABObj<vec, LAVector, double>(obj, double(1.)/f);
}

inline ABObj<vec,LAVector,double> operator-(const LAVector& obj) {
  return ABObj<vec,LAVector,double>(obj, double(-1.));
}
*/

#endif //AB_ABObj_H_
