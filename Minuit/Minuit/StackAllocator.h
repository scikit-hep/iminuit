#ifndef StackAllocator_H
#define StackAllocator_H

#include "Minuit/MnConfig.h"

// comment out this line and recompile if you want to gain additional 
// performance (the gain is mainly for "simple" functions which are easy
// to calculate and vanishes quickly if going to cost-intensive functions)
// the library is no longer thread save however 

// #define _MN_NO_THREAD_SAVE_

//#include <iostream>



#include <cstdlib>


/// define stack allocator symbol
 


class StackOverflow {};
class StackError {};
//  using namespace std;

/** StackAllocator controls the memory allocation/deallocation of Minuit. If
    _MN_NO_THREAD_SAVE_ is defined, memory is taken from a pre-allocated piece
    of heap memory which is then used like a stack, otherwise via standard
    malloc/free. Note that defining _MN_NO_THREAD_SAVE_ makes the code thread-
    unsave. The gain in performance is mainly for cost-cheap FCN functions.
 */

class StackAllocator {

public:

//   enum {default_size = 1048576};
  enum {default_size = 524288};

  StackAllocator() {
#ifdef _MN_NO_THREAD_SAVE_
    //std::cout<<"StackAllocator allocate "<<default_size<<std::endl;
    theStack = new unsigned char[default_size];
#endif
    theStackOffset = 0;
    theBlockCount = 0;
  }

  ~StackAllocator() {
#ifdef _MN_NO_THREAD_SAVE_
    //std::cout<<"StackAllocator destruct "<<theStackOffset<<std::endl;
    if(theStack) delete [] theStack;
#endif
  }

  void* allocate( size_t nBytes) {
#ifdef _MN_NO_THREAD_SAVE_
    if(theStack == 0) theStack = new unsigned char[default_size];
      int nAlloc = alignedSize(nBytes);
      checkOverflow(nAlloc);

//       std::cout << "Allocating " << nAlloc << " bytes, requested = " << nBytes << std::endl;

      // write the start position of the next block at the start of the block
      writeInt( theStackOffset, theStackOffset+nAlloc);
      // write the start position of the new block at the end of the block
      writeInt( theStackOffset + nAlloc - sizeof(int), theStackOffset);
 
      void* result = theStack + theStackOffset + sizeof(int);
      theStackOffset += nAlloc;
      theBlockCount++;

#ifdef DEBUG_ALLOCATOR
      checkConsistency();
#endif
      
#else
      void* result = malloc(nBytes);
#endif

      return result;
  }
  
  void deallocate( void* p) {
#ifdef _MN_NO_THREAD_SAVE_
      // int previousOffset = readInt( theStackOffset - sizeof(int));
      int delBlock = toInt(p);
      int nextBlock = readInt( delBlock);
      int previousBlock = readInt( nextBlock - sizeof(int));
      if ( nextBlock == theStackOffset) { 
          // deallocating last allocated
	  theStackOffset = previousBlock;
      }
      else {
          // overwrite previous adr of next block
	  int nextNextBlock = readInt(nextBlock);
	  writeInt( nextNextBlock - sizeof(int), previousBlock); 
	  // overwrite head of deleted block
	  writeInt( previousBlock, nextNextBlock);
      }
      theBlockCount--;

#ifdef DEBUG_ALLOCATOR
      checkConsistency();
#endif
#else
      free(p);
#endif
      // cout << "Block at " << delBlock 
      //   << " deallocated, theStackOffset = " << theStackOffset << endl;
  }

  int readInt( int offset) {
      int* ip = (int*)(theStack+offset);

      // cout << "read " << *ip << " from offset " << offset << endl;

      return *ip;
  }

  void writeInt( int offset, int value) {

      // cout << "writing " << value << " to offset " << offset << endl;

      int* ip = reinterpret_cast<int*>(theStack+offset);
      *ip = value;
  }

  int toInt( void* p) {
      unsigned char* pc = static_cast<unsigned char*>(p);

      // cout << "toInt: p = " << p << " theStack = " << (void*) theStack << endl;
	  // VC 7.1 warning:conversin from __w64 int to int
      int userBlock = pc - theStack;
      return userBlock - sizeof(int); // correct for starting int
  }

  int alignedSize( int nBytes) {
      const int theAlignment = 4;
      int needed = nBytes % theAlignment == 0 ? nBytes : (nBytes/theAlignment+1)*theAlignment;
      return needed + 2*sizeof(int);
  }

  void checkOverflow( int n) {
      if (theStackOffset + n >= default_size) {
	//std::cout << " no more space on stack allocator" << std::endl;
	  throw StackOverflow();
      }
  }

  bool checkConsistency() {

    //std::cout << "checking consistency for " << theBlockCount << " blocks"<< std::endl;

      // loop over all blocks
      int beg = 0;
      int end = theStackOffset;
      int nblocks = 0;
      while (beg < theStackOffset) {
	  end = readInt( beg);

	  // cout << "beg = " << beg << " end = " << end 
	  //     << " theStackOffset = " << theStackOffset << endl;

	  int beg2 = readInt( end - sizeof(int));
	  if ( beg != beg2) {
	    //std::cout << "  beg != beg2 " << std::endl;
	      return false;
	  }
	  nblocks++;
	  beg = end;
      }
      if (end != theStackOffset) {
	//std::cout << " end != theStackOffset" << std::endl;
	  return false;
      }
      if (nblocks != theBlockCount) {
	//std::cout << "nblocks != theBlockCount" << std::endl;
	  return false;
      }
      //std::cout << "Allocator is in consistent state, nblocks = " << nblocks << std::endl;
      return true;
  }

private:

  unsigned char* theStack;
//   unsigned char theStack[default_size];
  int            theStackOffset;
  int            theBlockCount;

};



class StackAllocatorHolder { 
  
  // t.b.d need to use same trick as  Boost singleton.hpp to be sure that 
  // StackAllocator is created before main() 

 public: 

    
  static StackAllocator & get() { 
    static StackAllocator gStackAllocator; 
    return gStackAllocator; 
  }
}; 



#endif
