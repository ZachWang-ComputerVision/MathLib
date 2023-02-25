#include <iostream>
#include <vector>
#include <list>
#include <cstdlib>


#include "NNLib.h"
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

  int za[2] = {2,2};
  size_t zs = 2;
  NNLib::Tensor<int> zero = NNLib::Zeros<int>(za, zs);
  std::cout << "zero matrix size: " << zero.matrix_size() << std::endl;

  NNLib::Tensor<int> zero_like = NNLib::Zeros_like<int>(zero);
  std::cout << "zero_like matrix size: " << zero_like.matrix_size() << std::endl;

  NNLib::Tensor<int> one = NNLib::Ones<int>(za, zs);
  std::cout << "one matrix size: " << one.matrix_size() << std::endl;
  
  NNLib::Tensor<float> rand_int = NNLib::Random_decimal<float>(za, zs);
  std::cout << "rand decimal matrix size: " << rand_int.matrix_size() << std::endl;
  std::cout << "rand decimal data: " << std::endl;
  for (size_t i = 0; i < rand_int.matrix_size(); i++) {
    std::cout << rand_int.data[i] << std::endl;
  };

  NNLib::Tensor<int> eye_one = NNLib::Eye<int>(za, zs);
  std::cout << "eye_one matrix size: " << eye_one.matrix_size() << std::endl;

  NNLib::Tensor<int> cat = NNLib::Concat<int>(zero, one, 0);
  std::cout << "cat matrix size: " << cat.matrix_size() << std::endl;

  return 0;
}



// g++ main.cpp -o main.exe MathLib.cpp