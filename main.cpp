#include <iostream>
#include <vector>
#include <list>
#include <cstdlib>


// #include "NNLib.h"
#include "Tensor.h"
// #include "test.cuh"

int main() {

  int a[] = {0,1,2,3,4,5};
  size_t as = 6;
  int b[2] = {2,3};
  size_t bs = 2;
  NNLib::Tensor<int> tensor(a, as, b, bs);
  std::cout << tensor.matrix_size() << std::endl;
  std::cout << tensor.shape_size() << std::endl;
  int nS[] = {6,1};
  size_t u = 2;
  tensor.reshape(nS, u);
  std::cout << tensor.matrix_size() << std::endl;
  std::cout << tensor.shape_size() << std::endl;
  int* dim = tensor.matrix_shape();
  std::cout << "---------------" << std::endl;
  std::cout << dim << std::endl;
  std::cout << dim[1] << std::endl;

  std::cout << "---------------" << std::endl;
  for (size_t i = 0; i < as; i++) {
    std::cout << a[i] << std::endl;
  };
  


  // Wrapper::wrapper();

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