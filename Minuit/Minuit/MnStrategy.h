#ifndef MN_MnStrategy_H_
#define MN_MnStrategy_H_

/** API class for defining three levels of strategies: low (0), medium (1), 
    high (>=2);
    acts on: Migrad (behavioural), 
             Minos (lowers strategy by 1 for Minos-own minimization), 
	     Hesse (iterations), 
	     Numerical2PDerivative (iterations)
 */

class MnStrategy {

public:

  //default strategy
  MnStrategy();

  //user defined strategy (0, 1, >=2)
  explicit MnStrategy(unsigned int);

  ~MnStrategy() {}

  unsigned int strategy() const {return theStrategy;}

  unsigned int gradientNCycles() const {return theGradNCyc;}
  double gradientStepTolerance() const {return theGradTlrStp;}
  double gradientTolerance() const {return theGradTlr;}

  unsigned int hessianNCycles() const {return theHessNCyc;}
  double hessianStepTolerance() const {return theHessTlrStp;}
  double hessianG2Tolerance() const {return theHessTlrG2;}
  unsigned int hessianGradientNCycles() const {return theHessGradNCyc;}
  
  bool isLow() const {return theStrategy == 0;}
  bool isMedium() const {return theStrategy == 1;}
  bool isHigh() const {return theStrategy >= 2;}

  void setLowStrategy();
  void setMediumStrategy();
  void setHighStrategy();
  
  void setGradientNCycles(unsigned int n) {theGradNCyc = n;}
  void setGradientStepTolerance(double stp) {theGradTlrStp = stp;}
  void setGradientTolerance(double toler) {theGradTlr = toler;}

  void setHessianNCycles(unsigned int n) {theHessNCyc = n;}
  void setHessianStepTolerance(double stp) {theHessTlrStp = stp;}
  void setHessianG2Tolerance(double toler) {theHessTlrG2 = toler;}
  void setHessianGradientNCycles(unsigned int n) {theHessGradNCyc = n;}
  
private:

  unsigned int theStrategy;

  unsigned int theGradNCyc;
  double theGradTlrStp;
  double theGradTlr;
  unsigned int theHessNCyc;
  double theHessTlrStp;
  double theHessTlrG2;
  unsigned int theHessGradNCyc;
};

#endif //MN_MnStrategy_H_
