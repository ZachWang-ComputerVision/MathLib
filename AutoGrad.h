#include <iostream>
#include <vector>
#include <list>

#include "NNLib.h"


class Tensor{
  public:
    Tensor(NNLib::Mat<std::vector<float>> data, bool grad, std::list<int> dependencies) {
      
    };

    ~Tensor() { };
};