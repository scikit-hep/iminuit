#ifndef MN_MnVectorTransform_H_
#define MN_MnVectorTransform_H_

#include "Minuit/MnMatrix.h"

class MnVectorTransform {

public:

  MnVectorTransform() {}

  ~MnVectorTransform() {}

  std::vector<double> operator()(const MnAlgebraicVector& avec) const {

    std::vector<double> result; result.reserve(avec.size());

    for(unsigned int i = 0; i < avec.size(); i++) result.push_back(avec(i));
    
    return result;
  }
  
};

#endif //MN_MnVectorTransform_H_
