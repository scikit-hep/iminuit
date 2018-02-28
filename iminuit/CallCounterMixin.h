#ifndef IMINUIT_CALLCOUNTERMIXIN_H
#define IMINUIT_CALLCOUNTERMIXIN_H

class CallCounterMixin {
public:
    CallCounterMixin(int n = 0) : ncall(n) {}
    int getNumCall() const { return ncall; }
    void resetNumCall() { ncall = 0; }
    void increaseNumCall() const { ++ncall; }
private:
    mutable int ncall;
};

#endif
