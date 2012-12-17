#ifndef MN_MnTimer_H_
#define MN_MnTimer_H_

//  
//  Vincenzo's PentiumTimer, taken from COBRA and adapted
//
//   V 0.0 
//

extern "C" inline unsigned long long int  rdtscPentium() {
  unsigned long long int x;
  __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
  return x;
}

class MnTimer {

public:
  
  typedef unsigned long long int PentiumTimeType;
  typedef long long int PentiumTimeIntervalType;
  typedef PentiumTimeType TimeType;

  inline static TimeType time() {return rdtscPentium();}

private:

};

#endif // MN_MnTimer_H_

