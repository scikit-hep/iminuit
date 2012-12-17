#ifndef MN_MnParabolaFactory_H_
#define MN_MnParabolaFactory_H_

class MnParabola;
class MnParabolaPoint;

class MnParabolaFactory {

public:

  MnParabolaFactory() {}

  ~MnParabolaFactory() {}

  MnParabola operator()(const MnParabolaPoint&, const MnParabolaPoint&, 
			const MnParabolaPoint&) const;

  MnParabola operator()(const MnParabolaPoint&, double, 
			const MnParabolaPoint&) const;

private: 
  
};

#endif //MN_MnParabolaFactory_H_
