#include <iostream>
#include <vector>
#include <list>

#include "MathLib.h"


class Tensor{
  public:
    Tensor(MathLib::Mat<std::vector<float>> data, bool grad, std::list<int> dependencies) {
      
    };

    ~Tensor() { };
};