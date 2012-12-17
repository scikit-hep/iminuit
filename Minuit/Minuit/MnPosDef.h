#ifndef MN_MnPosDef_H_
#define MN_MnPosDef_H_

class MinimumState;
class MinimumError;
class MnMachinePrecision;

class MnPosDef {

public:
  
  MnPosDef() {}
  
  ~MnPosDef() {}
  
  MinimumState operator()(const MinimumState&, const MnMachinePrecision&) const;
  MinimumError operator()(const MinimumError&, const MnMachinePrecision&) const;
private:

};

#endif //MN_MnPosDef_H_
