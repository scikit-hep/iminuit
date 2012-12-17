#ifndef MN_MnRefCountedPointer_H_
#define MN_MnRefCountedPointer_H_

#include "MnReferenceCounter.h"

template<class T> class MnRefCountedPointer {

public:

  // Default constructor needed for use inside array, vector, etc.
  MnRefCountedPointer() : thePtr(0), theCounter(0) {}
  
  MnRefCountedPointer(T* pt) : 
    thePtr(pt), theCounter(new MnReferenceCounter()) {addReference();}

  MnRefCountedPointer(const MnRefCountedPointer<T>& other) : 
    thePtr(other.thePtr), theCounter(other.theCounter) {addReference();}

  ~MnRefCountedPointer() {
    /*
    if(references() == 0) {
      if(thePtr) delete thePtr; 
      if(theCounter) delete theCounter;
    }
    else removeReference();
    */
    if(references() != 0) removeReference();
  }
  
  bool isValid() const {return thePtr != 0;}
  
  MnRefCountedPointer& operator=(const MnRefCountedPointer<T>& other) {
    if(thePtr != other.thePtr) {
      removeReference();
      thePtr = other.thePtr;
      theCounter = other.theCounter;
      addReference();
    }
    return *this;
  }

  MnRefCountedPointer& operator=(T* ptr) {
    if(thePtr != ptr) {
      thePtr = ptr;
      theCounter = new MnReferenceCounter();
    }
    return *this;
  }

  T* get() const {return thePtr;}

  T* operator->() const {check(); return thePtr;}
  
  T& operator*() const {check(); return *thePtr;}
  
  bool operator==(const  T* otherP) const {return thePtr == otherP;}
 
  bool operator<(const  T* otherP) const {return thePtr < otherP;}
 
  unsigned int references() const {return theCounter->references();}

  void addReference() const {theCounter->addReference();}

  void removeReference() {
    theCounter->removeReference();
    if(references() == 0) {
      delete thePtr; thePtr=0; 
      delete theCounter; theCounter=0;
    }
  }
  
private:
  
  T*  thePtr;  
  MnReferenceCounter* theCounter;
 
private:

  void check() const {assert(isValid());}  
};

#endif //MN_MnRefCountedPointer_H_
