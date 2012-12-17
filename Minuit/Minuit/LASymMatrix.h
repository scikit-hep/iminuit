#ifndef LA_LASymMatrix_H_
#define LA_LASymMatrix_H_

#include "Minuit/MnConfig.h"
#include "Minuit/ABSum.h"
#include "Minuit/VectorOuterProduct.h"
#include "Minuit/MatrixInverse.h"

#include <cassert>
#include <memory>


// #include <iostream>

#include "Minuit/StackAllocator.h"
//extern StackAllocator StackAllocatorHolder::get();

// for memcopy
#include <string.h>

int mndaxpy(unsigned int, double, const double*, int, double*, int);
int mndscal(unsigned int, double, double*, int);

class LAVector;

int invert ( LASymMatrix & );

class LASymMatrix {

private:

  LASymMatrix() : theSize(0), theNRow(0), theData(0) {}

public:

  typedef sym Type;

  LASymMatrix(unsigned int n) : theSize(n*(n+1)/2), theNRow(n), theData((double*)StackAllocatorHolder::get().allocate(sizeof(double)*n*(n+1)/2)) {
//     assert(theSize>0);
    memset(theData, 0, theSize*sizeof(double));
//     std::cout<<"LASymMatrix(unsigned int n), n= "<<n<<std::endl;
  }

  ~LASymMatrix() {
//     std::cout<<"~LASymMatrix()"<<std::endl;
//     if(theData) std::cout<<"deleting "<<theSize<<std::endl;
//     else std::cout<<"no delete"<<std::endl;
//     if(theData) delete [] theData;
    if(theData) StackAllocatorHolder::get().deallocate(theData);
  }

  LASymMatrix(const LASymMatrix& v) : 
    theSize(v.size()), theNRow(v.nrow()), theData((double*)StackAllocatorHolder::get().allocate(sizeof(double)*v.size())) {
//     std::cout<<"LASymMatrix(const LASymMatrix& v)"<<std::endl;
    memcpy(theData, v.data(), theSize*sizeof(double));
  }

  LASymMatrix& operator=(const LASymMatrix& v) {
//     std::cout<<"LASymMatrix& operator=(const LASymMatrix& v)"<<std::endl;
//     std::cout<<"theSize= "<<theSize<<std::endl;
//     std::cout<<"v.size()= "<<v.size()<<std::endl;
    assert(theSize == v.size());
    memcpy(theData, v.data(), theSize*sizeof(double));
    return *this;
  }

  template<class T>
  LASymMatrix(const ABObj<sym, LASymMatrix, T>& v) : 
    theSize(v.obj().size()), theNRow(v.obj().nrow()), theData((double*)StackAllocatorHolder::get().allocate(sizeof(double)*v.obj().size())) {
//     std::cout<<"LASymMatrix(const ABObj<sym, LASymMatrix, T>& v)"<<std::endl;
    //std::cout<<"allocate "<<theSize<<std::endl;    
    memcpy(theData, v.obj().data(), theSize*sizeof(double));
    mndscal(theSize, double(v.f()), theData, 1);
    //std::cout<<"theData= "<<theData[0]<<" "<<theData[1]<<std::endl;
  } 

  template<class A, class B, class T>
  LASymMatrix(const ABObj<sym, ABSum<ABObj<sym, A, T>, ABObj<sym, B, T> >,T>& sum) : theSize(0), theNRow(0), theData(0) {
//     std::cout<<"template<class A, class B, class T> LASymMatrix(const ABObj<sym, ABSum<ABObj<sym, A, T>, ABObj<sym, B, T> > >& sum)"<<std::endl;
//     recursive construction
    (*this) = sum.obj().a();
    (*this) += sum.obj().b();
    //std::cout<<"leaving template<class A, class B, class T> LASymMatrix(const ABObj..."<<std::endl;
  }

  template<class A, class T>
  LASymMatrix(const ABObj<sym, ABSum<ABObj<sym, LASymMatrix, T>, ABObj<sym, A, T> >,T>& sum) : theSize(0), theNRow(0), theData(0) {
//     std::cout<<"template<class A, class T> LASymMatrix(const ABObj<sym, ABSum<ABObj<sym, LASymMatrix, T>, ABObj<sym, A, T> >,T>& sum)"<<std::endl;

    // recursive construction
    //std::cout<<"(*this)=sum.obj().b();"<<std::endl;
    (*this)=sum.obj().b();
    //std::cout<<"(*this)+=sum.obj().a();"<<std::endl;
    (*this)+=sum.obj().a();  
    //std::cout<<"leaving template<class A, class T> LASymMatrix(const ABObj<sym, ABSum<ABObj<sym, LASymMatrix,.."<<std::endl;
  }

  template<class A, class T>
  LASymMatrix(const ABObj<sym, ABObj<sym, A, T>, T>& something) : theSize(0), theNRow(0), theData(0) {
//     std::cout<<"template<class A, class T> LASymMatrix(const ABObj<sym, ABObj<sym, A, T>, T>& something)"<<std::endl;
    (*this) = something.obj();
    (*this) *= something.f();
    //std::cout<<"leaving template<class A, class T> LASymMatrix(const ABObj<sym, ABObj<sym, A, T>, T>& something)"<<std::endl;
  }

  template<class T>
  LASymMatrix(const ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, T>, T>, T>& inv) : theSize(inv.obj().obj().obj().size()), theNRow(inv.obj().obj().obj().nrow()), theData((double*)StackAllocatorHolder::get().allocate(sizeof(double)*inv.obj().obj().obj().size())) {
    memcpy(theData, inv.obj().obj().obj().data(), theSize*sizeof(double));
    mndscal(theSize, double(inv.obj().obj().f()), theData, 1);
    invert(*this);
    mndscal(theSize, double(inv.f()), theData, 1);
  }

  template<class A, class T>
  LASymMatrix(const ABObj<sym, ABSum<ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, T>, T>, T>, ABObj<sym, A, T> >, T>& sum) : theSize(0), theNRow(0), theData(0) {
//     std::cout<<"template<class A, class T> LASymMatrix(const ABObj<sym, ABSum<ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, T>, T>, T>, ABObj<sym, A, T> >, T>& sum)"<<std::endl;

    // recursive construction
    (*this)=sum.obj().b();
    (*this)+=sum.obj().a();  
    //std::cout<<"leaving template<class A, class T> LASymMatrix(const ABObj<sym, ABSum<ABObj<sym, LASymMatrix,.."<<std::endl;
  }

  LASymMatrix(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, double>, double>, double>&);

  template<class A, class T>
  LASymMatrix(const ABObj<sym, ABSum<ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T>, ABObj<sym, A, T> >, T>& sum) : theSize(0), theNRow(0), theData(0) {
//     std::cout<<"template<class A, class T> LASymMatrix(const ABObj<sym, ABSum<ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T> ABObj<sym, A, T> >,T>& sum)"<<std::endl;

    // recursive construction
    (*this)=sum.obj().b();
    (*this)+=sum.obj().a();  
    //std::cout<<"leaving template<class A, class T> LASymMatrix(const ABObj<sym, ABSum<ABObj<sym, LASymMatrix,.."<<std::endl;
  }

  LASymMatrix& operator+=(const LASymMatrix& m) {
//     std::cout<<"LASymMatrix& operator+=(const LASymMatrix& m)"<<std::endl;
    assert(theSize==m.size());
    mndaxpy(theSize, 1., m.data(), 1, theData, 1);
    return *this;
  }

  LASymMatrix& operator-=(const LASymMatrix& m) {
//     std::cout<<"LASymMatrix& operator-=(const LASymMatrix& m)"<<std::endl;
    assert(theSize==m.size());
    mndaxpy(theSize, -1., m.data(), 1, theData, 1);
    return *this;
  }

  template<class T>
  LASymMatrix& operator+=(const ABObj<sym, LASymMatrix, T>& m) {
//     std::cout<<"template<class T> LASymMatrix& operator+=(const ABObj<sym, LASymMatrix, T>& m)"<<std::endl;
    assert(theSize==m.obj().size());
    if(m.obj().data()==theData) {
      mndscal(theSize, 1.+double(m.f()), theData, 1);
    } else {
      mndaxpy(theSize, double(m.f()), m.obj().data(), 1, theData, 1);
    }
    //std::cout<<"theData= "<<theData[0]<<" "<<theData[1]<<std::endl;
    return *this;
  }

  template<class A, class T>
  LASymMatrix& operator+=(const ABObj<sym, A, T>& m) {
//     std::cout<<"template<class A, class T> LASymMatrix& operator+=(const ABObj<sym, A,T>& m)"<<std::endl;
    (*this) += LASymMatrix(m);
    return *this;
  }

  template<class T>
  LASymMatrix& operator+=(const ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, T>, T>, T>& m) {
//     std::cout<<"template<class T> LASymMatrix& operator+=(const ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, T>, T>, T>& m)"<<std::endl;
    assert(theNRow > 0);
    LASymMatrix tmp(m.obj().obj());
    invert(tmp);
    tmp *= double(m.f());
    (*this) += tmp;
    return *this;
  }

  template<class T>
  LASymMatrix& operator+=(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T>& m) {
//     std::cout<<"template<class T> LASymMatrix& operator+=(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T>&"<<std::endl;
    assert(theNRow > 0);
    outer_prod(*this, m.obj().obj().obj(), m.f()*m.obj().obj().f()*m.obj().obj().f());
    return *this;
  }
  
  LASymMatrix& operator*=(double scal) {
    mndscal(theSize, scal, theData, 1);
    return *this;
  }

  double operator()(unsigned int row, unsigned int col) const {
    assert(row<theNRow && col < theNRow);
    if(row > col) 
      return theData[col+row*(row+1)/2];
    else
      return theData[row+col*(col+1)/2];
  }

  double& operator()(unsigned int row, unsigned int col) {
    assert(row<theNRow && col < theNRow);
    if(row > col) 
      return theData[col+row*(row+1)/2];
    else
      return theData[row+col*(col+1)/2];
  }
  
  const double* data() const {return theData;}

  double* data() {return theData;}
  
  unsigned int size() const {return theSize;}

  unsigned int nrow() const {return theNRow;}
  
  unsigned int ncol() const {return nrow();}

private:
 
  unsigned int theSize;
  unsigned int theNRow;
  double* theData;

public:

  template<class T>
  LASymMatrix& operator=(const ABObj<sym, LASymMatrix, T>& v)  {
    //std::cout<<"template<class T> LASymMatrix& operator=(ABObj<sym, LASymMatrix, T>& v)"<<std::endl;
    if(theSize == 0 && theData == 0) {
      theSize = v.obj().size();
      theNRow = v.obj().nrow();
      theData = (double*)StackAllocatorHolder::get().allocate(sizeof(double)*theSize);
    } else {
      assert(theSize == v.obj().size());
    }
    //std::cout<<"theData= "<<theData[0]<<" "<<theData[1]<<std::endl;
    memcpy(theData, v.obj().data(), theSize*sizeof(double));
    (*this) *= v.f();
    return *this;
  }

  template<class A, class T>
  LASymMatrix& operator=(const ABObj<sym, ABObj<sym, A, T>, T>& something) {
    //std::cout<<"template<class A, class T> LASymMatrix& operator=(const ABObj<sym, ABObj<sym, A, T>, T>& something)"<<std::endl;
    if(theSize == 0 && theData == 0) {
      (*this) = something.obj();
      (*this) *= something.f();
    } else {
      LASymMatrix tmp(something.obj());
      tmp *= something.f();
      assert(theSize == tmp.size());
      memcpy(theData, tmp.data(), theSize*sizeof(double)); 
    }
    //std::cout<<"template<class A, class T> LASymMatrix& operator=(const ABObj<sym, ABObj<sym, A, T>, T>& something)"<<std::endl;
    return *this;
  }

  template<class A, class B, class T>
  LASymMatrix& operator=(const ABObj<sym, ABSum<ABObj<sym, A, T>, ABObj<sym, B, T> >,T>& sum) {
    //std::cout<<"template<class A, class B, class T> LASymMatrix& operator=(const ABObj<sym, ABSum<ABObj<sym, A, T>, ABObj<sym, B, T> >,T>& sum)"<<std::endl;
    // recursive construction
    if(theSize == 0 && theData == 0) {
      (*this) = sum.obj().a();
      (*this) += sum.obj().b();
      (*this) *= sum.f();
    } else {
      LASymMatrix tmp(sum.obj().a());
      tmp += sum.obj().b();
      tmp *= sum.f();
      assert(theSize == tmp.size());
      memcpy(theData, tmp.data(), theSize*sizeof(double));
    }
    return *this;
  }

  template<class A, class T>
  LASymMatrix& operator=(const ABObj<sym, ABSum<ABObj<sym, LASymMatrix, T>, ABObj<sym, A, T> >,T>& sum)  {
    //std::cout<<"template<class A, class T> LASymMatrix& operator=(const ABObj<sym, ABSum<ABObj<sym, LASymMatrix, T>, ABObj<sym, A, T> >,T>& sum)"<<std::endl;
    
    if(theSize == 0 && theData == 0) {
      //std::cout<<"theSize == 0 && theData == 0"<<std::endl;
      (*this) = sum.obj().b();
      (*this) += sum.obj().a();
      (*this) *= sum.f();
    } else {
      //std::cout<<"creating tmp variable"<<std::endl;
      LASymMatrix tmp(sum.obj().b());
      tmp += sum.obj().a();
      tmp *= sum.f();
      assert(theSize == tmp.size());
      memcpy(theData, tmp.data(), theSize*sizeof(double));
    }
    //std::cout<<"leaving LASymMatrix& operator=(const ABObj<sym, ABSum<ABObj<sym, LASymMatrix..."<<std::endl;
    return *this;
  }

  template<class T>
  LASymMatrix& operator=(const ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, T>, T>, T>& inv) {
    if(theSize == 0 && theData == 0) {
      theSize = inv.obj().obj().obj().size();
      theNRow = inv.obj().obj().obj().nrow();
      theData = (double*)StackAllocatorHolder::get().allocate(sizeof(double)*theSize);
      memcpy(theData, inv.obj().obj().obj().data(), theSize*sizeof(double));
      (*this) *= inv.obj().obj().f();
      invert(*this);
      (*this) *= inv.f();
    } else {
      LASymMatrix tmp(inv.obj().obj());
      invert(tmp);
      tmp *= double(inv.f());
      assert(theSize == tmp.size());
      memcpy(theData, tmp.data(), theSize*sizeof(double));
    }
    return *this;
  }

  LASymMatrix& operator=(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, double>, double>, double>&);
};

#endif //LA_LASymMatrix_H_
