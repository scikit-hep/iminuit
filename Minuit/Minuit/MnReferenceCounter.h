#ifndef MN_MnReferenceCounter_H_
#define MN_MnReferenceCounter_H_

#include <cassert>

#include "StackAllocator.h"

//extern StackAllocator gStackAllocator;

class MnReferenceCounter {

public:

  MnReferenceCounter() : theReferences(0) {}

  MnReferenceCounter(const MnReferenceCounter& other) : 
    theReferences(other.theReferences) {}

  MnReferenceCounter& operator=(const MnReferenceCounter& other) {
    theReferences = other.theReferences;
    return *this;
  }
  
  ~MnReferenceCounter() {assert(theReferences == 0);}
  
  void* operator new(size_t nbytes) {
    return StackAllocatorHolder::get().allocate(nbytes);
  }
  
  void operator delete(void* p, size_t /*nbytes */) {
    StackAllocatorHolder::get().deallocate(p);
  }

  unsigned int references() const {return theReferences;}

  void addReference() const {theReferences++;}

  void removeReference() const {theReferences--;}
  
private:
  
  mutable unsigned int theReferences;
};

#endif //MN_MnReferenceCounter_H_
