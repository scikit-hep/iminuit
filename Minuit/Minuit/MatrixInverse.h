#ifndef AB_MatrixInverse_H_
#define AB_MatrixInverse_H_

#include "Minuit/ABTypes.h"
#include "Minuit/ABObj.h"

template<class mtype, class M, class T>
class MatrixInverse {

public:

  MatrixInverse(const M& obj) : theObject(obj) {}

  ~MatrixInverse() {}

  typedef mtype Type;

  const M& obj() const {return theObject;}

private:

  M theObject;
};

template<class M, class T>
class MatrixInverse<vec, M, T> {

private:

  MatrixInverse(const M& obj) : theObject(obj) {}

public:

  ~MatrixInverse() {}

  typedef vec Type;

  const M& obj() const {return theObject;}

private:

  M theObject;
};

template<class mt, class M, class T>
inline ABObj<mt, MatrixInverse<mt, ABObj<mt, M, T>, T>, T> inverse(const ABObj<mt, M, T>& obj) {
  return ABObj<mt, MatrixInverse<mt, ABObj<mt, M, T>, T>, T>(MatrixInverse<mt, ABObj<mt, M, T>, T>(obj));
}

#endif //AB_MatrixInverse_H_
