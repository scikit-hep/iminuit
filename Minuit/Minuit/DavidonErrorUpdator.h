#ifndef MN_DavidonErrorUpdator_H_
#define MN_DavidonErrorUpdator_H_

#include "Minuit/MinimumErrorUpdator.h"

class DavidonErrorUpdator : public MinimumErrorUpdator {

public:

  DavidonErrorUpdator() {}
  
  virtual ~DavidonErrorUpdator() {}

  virtual MinimumError update(const MinimumState&, const MinimumParameters&,
			      const FunctionGradient&) const;

private:

};

#endif //MN_DavidonErrorUpdator_H_
