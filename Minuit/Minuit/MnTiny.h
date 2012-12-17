#ifndef MN_MnTiny_H_
#define MN_MnTiny_H_

class MnTiny {
  
public:
  
  MnTiny() : theOne(1.) {}
  
  ~MnTiny() {}
  
  double one() const;
  
  double operator()(double epsp1) const;
  
private:
  
  double theOne;
};

#endif // MN_MnTiny_H_
