#include <iostream>
#include <vector>
#include <list>
#include <cstdlib>


// #include "NNLib.h"

void test(int t[]) {
  for (int i = 0; i < 5; i++) {
    std::cout << "in func: " << t[i] << std::endl;
  };
};

int main() {

  // std::vector<int> a = {1,2,3,4,5,6};
  // std::vector<int> b = {1,1,2,3};
  // std::vector<float> vec;
  // vec.reserve(10);
  // for (int i = 0; i < 10; i++) { vec.push_back( (float)(rand() % 100) / 100); };
  // for (int i = 0; i < 10; i++) {
  //   std::cout << vec[i] << ",";
  // }; std::cout << std::endl;

  // MathLib::Mat<std::vector<int>> mat(a, b);
  // MathLib::Mat<std::vector<int>> c = MathLib::zeros<int>(b);

  // mat.reshape(c);
  // std::vector<int> shape =  c.shape();
  // for (int i = 0; i < 4; i++) {
  //   std::cout << shape[i] << ",";
  // }; std::cout << std::endl;

  return 0;
}



// g++ main.cpp -o main.exe MathLib.cpp