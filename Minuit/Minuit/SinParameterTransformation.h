#ifndef MN_SinParameterTransformation_H_
#define MN_SinParameterTransformation_H_

class MnMachinePrecision;

class SinParameterTransformation {

public:

  SinParameterTransformation() {}

  ~SinParameterTransformation() {}

  double int2ext(double value, double upper, double lower) const;
  double ext2int(double value, double upper, double lower, 
		 const MnMachinePrecision&) const;
  double dInt2Ext(double value, double upper, double lower) const;

private:

};

#endif //MN_SinParameterTransformation_H_
