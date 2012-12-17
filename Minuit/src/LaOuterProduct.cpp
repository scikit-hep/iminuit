#include "Minuit/LaOuterProduct.h"
#include "Minuit/LAVector.h"
#include "Minuit/LASymMatrix.h"

int mndspr(const char*, unsigned int, double, const double*, int, double*);

LASymMatrix::LASymMatrix(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, double>, double>, double>& out) : theSize(0), theNRow(0), theData(0) {
//   std::cout<<"LASymMatrix::LASymMatrix(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, double>, double>, double>& out)"<<std::endl;
  theNRow = out.obj().obj().obj().size();
  theSize = theNRow*(theNRow+1)/2;
  theData = (double*)StackAllocatorHolder::get().allocate(sizeof(double)*theSize);
  memset(theData, 0, theSize*sizeof(double));
  outer_prod(*this, out.obj().obj().obj(), out.f()*out.obj().obj().f()*out.obj().obj().f());
}

LASymMatrix& LASymMatrix::operator=(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, double>, double>, double>& out) {
//   std::cout<<"LASymMatrix& LASymMatrix::operator=(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, double>, double>, double>& out)"<<std::endl;
  if(theSize == 0 && theData == 0) {
    theNRow = out.obj().obj().obj().size();
    theSize = theNRow*(theNRow+1)/2;
    theData = (double*)StackAllocatorHolder::get().allocate(sizeof(double)*theSize);
    memset(theData, 0, theSize*sizeof(double));
    outer_prod(*this, out.obj().obj().obj(), out.f()*out.obj().obj().f()*out.obj().obj().f());
  } else {
    LASymMatrix tmp(out.obj().obj().obj().size());
    outer_prod(tmp, out.obj().obj().obj());
    tmp *= double(out.f()*out.obj().obj().f()*out.obj().obj().f());
    assert(theSize == tmp.size());
    memcpy(theData, tmp.data(), theSize*sizeof(double));
  }
  return *this;
}

void outer_prod(LASymMatrix& A, const LAVector& v, double f) {

  mndspr("U", v.size(), f, v.data(), 1, A.data());
}
