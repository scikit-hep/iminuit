#ifndef LA_LAVector_H_
#define LA_LAVector_H_

#include "Minuit/ABSum.h"
#include "Minuit/ABProd.h"
#include "Minuit/LASymMatrix.h"

#include <cassert>
#include <memory>
// #include <iostream>

#include "Minuit/StackAllocator.h"
//extern StackAllocator StackAllocatorHolder::get();

int mndaxpy(unsigned int, double, const double*, int, double*, int);
int mndscal(unsigned int, double, double*, int);
int mndspmv(const char*, unsigned int, double, const double*, const double*, int, double, double*, int);

class LAVector {

private:

  LAVector() : theSize(0), theData(0) {}

public:

  typedef vec Type;

//   LAVector() : theSize(0), theData(0) {}

  LAVector(unsigned int n) : theSize(n), theData((double*)StackAllocatorHolder::get().allocate(sizeof(double)*n)) {
//     assert(theSize>0);
    memset(theData, 0, size()*sizeof(double));
//     std::cout<<"LAVector(unsigned int n), n= "<<n<<std::endl;
  }

  ~LAVector() {
//     std::cout<<"~LAVector()"<<std::endl;
//    if(theData) std::cout<<"deleting "<<theSize<<std::endl;
//     else std::cout<<"no delete"<<std::endl;
//     if(theData) delete [] theData;
    if(theData) StackAllocatorHolder::get().deallocate(theData);
  }

  LAVector(const LAVector& v) : 
    theSize(v.size()), theData((double*)StackAllocatorHolder::get().allocate(sizeof(double)*v.size())) {
//     std::cout<<"LAVector(const LAVector& v)"<<std::endl;
    memcpy(theData, v.data(), theSize*sizeof(double));
  }

  LAVector& operator=(const LAVector& v) {
//     std::cout<<"LAVector& operator=(const LAVector& v)"<<std::endl;
//     std::cout<<"theSize= "<<theSize<<std::endl;
//     std::cout<<"v.size()= "<<v.size()<<std::endl;
    assert(theSize == v.size());
    memcpy(theData, v.data(), theSize*sizeof(double));
    return *this;
  }

  template<class T>
  LAVector(const ABObj<vec, LAVector, T>& v) : 
    theSize(v.obj().size()), theData((double*)StackAllocatorHolder::get().allocate(sizeof(double)*v.obj().size())) {
//     std::cout<<"LAVector(const ABObj<LAVector, T>& v)"<<std::endl;
//     std::cout<<"allocate "<<theSize<<std::endl;    
    memcpy(theData, v.obj().data(), theSize*sizeof(T));
    (*this) *= T(v.f());
//     std::cout<<"theData= "<<theData[0]<<" "<<theData[1]<<std::endl;
  } 

  template<class A, class B, class T>
  LAVector(const ABObj<vec, ABSum<ABObj<vec, A, T>, ABObj<vec, B, T> >,T>& sum) : theSize(0), theData(0) {
//     std::cout<<"template<class A, class B, class T> LAVector(const ABObj<ABSum<ABObj<A, T>, ABObj<B, T> > >& sum)"<<std::endl;
    (*this) = sum.obj().a();
    (*this) += sum.obj().b();
    (*this) *= double(sum.f());
  }

  template<class A, class T>
  LAVector(const ABObj<vec, ABSum<ABObj<vec, LAVector, T>, ABObj<vec, A, T> >,T>& sum) : theSize(0), theData(0) {
//     std::cout<<"template<class A, class T> LAVector(const ABObj<ABSum<ABObj<LAVector, T>, ABObj<A, T> >,T>& sum)"<<std::endl;

    // recursive construction
//     std::cout<<"(*this)=sum.obj().b();"<<std::endl;
    (*this) = sum.obj().b();
//     std::cout<<"(*this)+=sum.obj().a();"<<std::endl;
    (*this) += sum.obj().a();  
    (*this) *= double(sum.f());
//     std::cout<<"leaving template<class A, class T> LAVector(const ABObj<ABSum<ABObj<LAVector,.."<<std::endl;
  }

  template<class A, class T>
  LAVector(const ABObj<vec, ABObj<vec, A, T>, T>& something) : theSize(0), theData(0) {
//     std::cout<<"template<class A, class T> LAVector(const ABObj<ABObj<A, T>, T>& something)"<<std::endl;
    (*this) = something.obj();
    (*this) *= something.f();
  }
  
  //
  template<class T>
  LAVector(const ABObj<vec, ABProd<ABObj<sym, LASymMatrix, T>, ABObj<vec, LAVector, T> >, T>& prod) : theSize(prod.obj().b().obj().size()), theData((double*)StackAllocatorHolder::get().allocate(sizeof(double)*prod.obj().b().obj().size())) {
//     std::cout<<"template<class T> LAVector(const ABObj<vec, ABProd<ABObj<sym, LASymMatrix, T>, ABObj<vec, LAVector, T> >, T>& prod)"<<std::endl;

    mndspmv("U", theSize, prod.f()*prod.obj().a().f()*prod.obj().b().f(), prod.obj().a().obj().data(), prod.obj().b().obj().data(), 1, 0., theData, 1);
  }

  //
  template<class T>
  LAVector(const ABObj<vec, ABSum<ABObj<vec, ABProd<ABObj<sym, LASymMatrix, T>, ABObj<vec, LAVector, T> >, T>, ABObj<vec, LAVector, T> >, T>& prod) : theSize(0), theData(0) {
    (*this) = prod.obj().b();
    (*this) += prod.obj().a();
    (*this) *= double(prod.f());    
  }

  //
  LAVector& operator+=(const LAVector& m) {
//     std::cout<<"LAVector& operator+=(const LAVector& m)"<<std::endl;
    assert(theSize==m.size());
    mndaxpy(theSize, 1., m.data(), 1, theData, 1);
    return *this;
  }

  LAVector& operator-=(const LAVector& m) {
//     std::cout<<"LAVector& operator-=(const LAVector& m)"<<std::endl;
    assert(theSize==m.size());
    mndaxpy(theSize, -1., m.data(), 1, theData, 1);
    return *this;
  }

  template<class T>
  LAVector& operator+=(const ABObj<vec, LAVector, T>& m) {
//     std::cout<<"template<class T> LAVector& operator+=(const ABObj<LAVector, T>& m)"<<std::endl;
    assert(theSize==m.obj().size());
    if(m.obj().data()==theData) {
      mndscal(theSize, 1.+double(m.f()), theData, 1);
    } else {
      mndaxpy(theSize, double(m.f()), m.obj().data(), 1, theData, 1);
    }
//     std::cout<<"theData= "<<theData[0]<<" "<<theData[1]<<std::endl;
    return *this;
  }

  template<class A, class T>
  LAVector& operator+=(const ABObj<vec, A, T>& m) {
//     std::cout<<"template<class A, class T> LAVector& operator+=(const ABObj<A,T>& m)"<<std::endl;
    (*this) += LAVector(m);
    return *this;
  }

  template<class T>
  LAVector& operator+=(const ABObj<vec, ABProd<ABObj<sym, LASymMatrix, T>, ABObj<vec, LAVector, T> >, T>& prod) {
    mndspmv("U", theSize, prod.f()*prod.obj().a().f()*prod.obj().b().f(), prod.obj().a().obj().data(), prod.obj().b().data(), 1, 1., theData, 1);
    return *this;
  }
  
  LAVector& operator*=(double scal) {
    mndscal(theSize, scal, theData, 1);
    return *this;
  }

  double operator()(unsigned int i) const {
    assert(i<theSize);
    return theData[i];
  }

  double& operator()(unsigned int i) {
//     std::cout<<"double& operator()(unsigned int i), i= "<<i<<std::endl;
    assert(i<theSize);
    return theData[i];
  }
  
  const double* data() const {return theData;}

  double* data() {return theData;}
  
  unsigned int size() const {return theSize;}
  
private:
 
  unsigned int theSize;
  double* theData;

public:

  template<class T>
  LAVector& operator=(const ABObj<vec, LAVector, T>& v)  {
//     std::cout<<"template<class T> LAVector& operator=(ABObj<LAVector, T>& v)"<<std::endl;
    if(theSize == 0 && theData == 0) {
      theSize = v.obj().size();
      theData = (double*)StackAllocatorHolder::get().allocate(sizeof(double)*theSize);
    } else {
      assert(theSize == v.obj().size());
    }
    memcpy(theData, v.obj().data(), theSize*sizeof(double));
    (*this) *= T(v.f());
    return *this;
  }

  template<class A, class T>
  LAVector& operator=(const ABObj<vec, ABObj<vec, A, T>, T>& something) {
//     std::cout<<"template<class A, class T> LAVector& operator=(const ABObj<ABObj<A, T>, T>& something)"<<std::endl;
    if(theSize == 0 && theData == 0) {
      (*this) = something.obj();
    } else {
      LAVector tmp(something.obj());
      assert(theSize == tmp.size());
      memcpy(theData, tmp.data(), theSize*sizeof(double)); 
    }
    (*this) *= something.f();
    return *this;
  }

  template<class A, class B, class T>
  LAVector& operator=(const ABObj<vec, ABSum<ABObj<vec, A, T>, ABObj<vec, B, T> >,T>& sum) {
    if(theSize == 0 && theData == 0) {
      (*this) = sum.obj().a();
      (*this) += sum.obj().b();
    } else {
      LAVector tmp(sum.obj().a());
      tmp += sum.obj().b();
      assert(theSize == tmp.size());
      memcpy(theData, tmp.data(), theSize*sizeof(double));
    }
    (*this) *= sum.f();
    return *this;
  }

  template<class A, class T>
  LAVector& operator=(const ABObj<vec, ABSum<ABObj<vec, LAVector, T>, ABObj<vec, A, T> >,T>& sum)  {
    if(theSize == 0 && theData == 0) {
      (*this) = sum.obj().b();
      (*this) += sum.obj().a();
    } else {
      LAVector tmp(sum.obj().a());
      tmp += sum.obj().b();
      assert(theSize == tmp.size());
      memcpy(theData, tmp.data(), theSize*sizeof(double));
    }
  (*this) *= sum.f();
    return *this;
  }

  //
  template<class T>
  LAVector& operator=(const ABObj<vec, ABProd<ABObj<sym, LASymMatrix, T>, ABObj<vec, LAVector, T> >, T>& prod) {
    if(theSize == 0 && theData == 0) {
      theSize = prod.obj().b().obj().size();
      theData = (double*)StackAllocatorHolder::get().allocate(sizeof(double)*theSize);
      mndspmv("U", theSize, double(prod.f()*prod.obj().a().f()*prod.obj().b().f()), prod.obj().a().obj().data(), prod.obj().b().obj().data(), 1, 0., theData, 1);    
    } else {
      LAVector tmp(prod.obj().b());
      assert(theSize == tmp.size());
      mndspmv("U", theSize, double(prod.f()*prod.obj().a().f()), prod.obj().a().obj().data(), tmp.data(), 1, 0., theData, 1);
    }      
    return *this;
  }

  //
  template<class T>
  LAVector& operator=(const ABObj<vec, ABSum<ABObj<vec, ABProd<ABObj<sym, LASymMatrix, T>, ABObj<vec, LAVector, T> >, T>, ABObj<vec, LAVector, T> >, T>& prod) {
    if(theSize == 0 && theData == 0) {
      (*this) = prod.obj().b();
      (*this) += prod.obj().a();
    } else {
      //       std::cout<<"creating tmp variable"<<std::endl;
      LAVector tmp(prod.obj().b());
      tmp += prod.obj().a();
      assert(theSize == tmp.size());
      memcpy(theData, tmp.data(), theSize*sizeof(double));
    }
    (*this) *= prod.f();
    return *this;
  }

};

#endif //LA_LAVector_H_
