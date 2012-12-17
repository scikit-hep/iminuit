#ifndef AB_ABTypes_H_
#define AB_ABTypes_H_

class gen {};
class sym {};
class vec {};

template<class A, class B>
class AlgebraicSumType {
public:
  typedef gen Type;
};

template<class T>
class AlgebraicSumType<T, T> {
public:
  typedef T Type;
};

template < >
class AlgebraicSumType<vec, gen> {
private:
  typedef gen Type;
};

template < >
class AlgebraicSumType<gen, vec> {
private:
  typedef gen Type;
};

template < >
class AlgebraicSumType<vec, sym> {
private:
  typedef gen Type;
};

template < >
class AlgebraicSumType<sym, vec> {
private:
  typedef gen Type;
};

//

template<class A, class B>
class AlgebraicProdType {
private:
  typedef gen Type;
};

template<class T>
class AlgebraicProdType<T, T> {
private:
  typedef T Type;
};

template < >
class AlgebraicProdType<gen, gen> {
public:
  typedef gen Type;
};

template < >
class AlgebraicProdType<sym, sym> {
public:
  typedef gen Type;
};

template < >
class AlgebraicProdType<sym, gen> {
public:
  typedef gen Type;
};

template < >
class AlgebraicProdType<gen, sym> {
public:
  typedef gen Type;
};

template < >
class AlgebraicProdType<vec, gen> {
private:
  typedef gen Type;
};

template < >
class AlgebraicProdType<gen, vec> {
public:
   typedef vec Type;
};

template < >
class AlgebraicProdType<vec, sym> {
private:
  typedef gen Type;
};

template < >
class AlgebraicProdType<sym, vec> {
public:
  typedef vec Type;
};



#endif //  AB_ABTypes_H_
