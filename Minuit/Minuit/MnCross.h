#ifndef MN_MnCross_H_
#define MN_MnCross_H_

#include "Minuit/MnUserParameterState.h"

class MnCross {

public:

    class CrossParLimit {};
    class CrossFcnLimit {};
    class CrossNewMin {};

public:

    MnCross() : theValue(0.), theState(MnUserParameterState()),
                theNFcn(0), theValid(false), theLimset(false),
                theMaxFcn(false), theNewMin(false) {}

    MnCross(unsigned int nfcn) : theValue(0.),
                                 theState(MnUserParameterState()),
                                 theNFcn(nfcn), theValid(false),
                                 theLimset(false), theMaxFcn(false),
                                 theNewMin(false) {}

    MnCross(const  MnUserParameterState& state, unsigned int nfcn) :
        theValue(0.), theState(state), theNFcn(nfcn), theValid(false),
        theLimset(false), theMaxFcn(false), theNewMin(false) {} 

    MnCross(double value, const MnUserParameterState& state, unsigned int nfcn):
        theValue(value), theState(state), theNFcn(nfcn), theValid(true),
        theLimset(false), theMaxFcn(false), theNewMin(false) {}

    MnCross(const MnUserParameterState& state, unsigned int nfcn, CrossParLimit):
        theValue(0.), theState(state), theNFcn(nfcn), theValid(false),
        theLimset(true), theMaxFcn(false), theNewMin(false) {}

    MnCross(const MnUserParameterState& state, unsigned int nfcn, CrossFcnLimit):
        theValue(0.), theState(state), theNFcn(nfcn), theValid(false),
        theLimset(false), theMaxFcn(true), theNewMin(false) {}

    MnCross(const MnUserParameterState& state, unsigned int nfcn, CrossNewMin) :
        theValue(0.), theState(state), theNFcn(nfcn), theValid(false),
        theLimset(false), theMaxFcn(false), theNewMin(true) {}

    ~MnCross() {}

    MnCross(const MnCross& cross) : theValue(cross.theValue),
        theState(cross.theState), theNFcn(cross.theNFcn),
        theValid(cross.theValid), theLimset(cross.theLimset),
        theMaxFcn(cross.theMaxFcn), theNewMin(cross.theNewMin) {}

    MnCross& operator()(const MnCross& cross) {
        theValue = cross.theValue;
        theState = cross.theState;
        theNFcn = cross.theNFcn;
        theValid = cross.theValid;
        theLimset = cross.theLimset;
        theMaxFcn = cross.theMaxFcn;
        theNewMin = cross.theNewMin;
        return *this;
    }

    double value() const {return theValue;}
    const MnUserParameterState& state() const {return theState;}
    bool isValid() const {return theValid;}
    bool atLimit() const {return theLimset;}
    bool atMaxFcn() const {return theMaxFcn;}
    bool newMinimum() const {return theNewMin;}
    unsigned int nfcn() const {return theNFcn;}

private:

    double theValue;
    MnUserParameterState theState;
    unsigned int theNFcn;
    bool theValid;
    bool theLimset;
    bool theMaxFcn;
    bool theNewMin;
};

#endif //MN_MnCross_H_
