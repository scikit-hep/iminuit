#include "Minuit/MnMachinePrecision.h"
#include "Minuit/MnTiny.h"

MnMachinePrecision::MnMachinePrecision() : theEpsMac(4.0E-7),
					   theEpsMa2(2.*sqrt(4.0E-7)) {
    
  //determine machine precision
  /*
  char e[] = {"e"};
  theEpsMac = 8.*dlamch_(e);
  theEpsMa2 = 2.*sqrt(theEpsMac);
  */

//   std::cout<<"machine precision eps: "<<eps()<<std::endl;
  
  MnTiny mytiny;
  
  //calculate machine precision
  double epstry = 0.5;
  double epsbak = 0.;
  double epsp1 = 0.;
  double one = 1.0;
  for(int i = 0; i < 100; i++) {
    epstry *= 0.5;
    epsp1 = one + epstry;
    epsbak = mytiny(epsp1);
    if(epsbak < epstry) {
      theEpsMac = 8.*epstry;
      theEpsMa2 = 2.*sqrt(theEpsMac);
      break;
    }
  } 
  
}
