#ifndef AB_VectorOuterProduct_H_
#define AB_VectorOuterProduct_H_

#include "Minuit/ABTypes.h"
#include "Minuit/ABObj.h"

template<class M, class T>
class VectorOuterProduct {

public:

  VectorOuterProduct(const M& obj) : theObject(obj) {}

  ~VectorOuterProduct() {}

  typedef sym Type;

  const M& obj() const {return theObject;}

private:

  M theObject;
};

template<class M, class T>
inline ABObj<sym, VectorOuterProduct<ABObj<vec, M, T>, T>, T> outer_product(const ABObj<vec, M, T>& obj) {
  return ABObj<sym, VectorOuterProduct<ABObj<vec, M, T>, T>, T>(VectorOuterProduct<ABObj<vec, M, T>, T>(obj));
}

#endif //AB_VectorOuterProduct_H_
