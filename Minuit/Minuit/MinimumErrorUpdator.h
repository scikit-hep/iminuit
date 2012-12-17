#ifndef MN_MinimumErrorUpdator_H_
#define MN_MinimumErrorUpdator_H_

class MinimumState;
class MinimumError;
class MinimumParameters;
class FunctionGradient;

class MinimumErrorUpdator {

public:

  virtual ~MinimumErrorUpdator() {}

  virtual MinimumError update(const MinimumState&, const MinimumParameters&,
			      const FunctionGradient&) const = 0;

};

#endif //MN_MinimumErrorUpdator_H_
