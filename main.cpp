#include <iostream>
#include <vector>
#include <list>
#include <cstdlib>
// #include <tbb/detail/

#include "NNLib.h"
// #include "test.cuh"


template <typename T> std::vector<T> test(T a, T b, T (*func)(T, T)) {
  std::vector<T> c;
  T d = func(a, b);
  c.push_back(d);
  return c;
};

int main() {

  std::vector<int> t = test<int>(5, 6, [](int a, int b){ return a + b; });
  std::cout << "t: "<< t[0] << std::endl;
  // tbb::pa


  int a[] = {0,1,2,3,4,5};
  size_t as = 6;

  // NNLib::DataContainer<int> ds {0,1,2,3,4,5};
  // std::cout << "ds: " << ds.size() << std::endl;

  
  int b[2] = {2,3};
  size_t bs = 2;
  NNLib::Tensor<int> tensor(a, as, b, bs);
  // std::cout << tensor.matrix_size() << std::endl;
  // std::cout << tensor.shape_size() << std::endl;
  int nS[] = {6,1};
  size_t u = 2;
  tensor.reshape(nS, u);
  // std::cout << tensor.matrix_size() << std::endl;
  // std::cout << tensor.shape_size() << std::endl;
  int* dim = tensor.matrix_shape();
  // std::cout << "---------------" << std::endl;
  // std::cout << dim << std::endl;
  // std::cout << dim[1] << std::endl;

  // std::cout << "---------------" << std::endl;
  // for (size_t i = 0; i < as; i++) {
  //   std::cout << a[i] << std::endl;
  // };

  int za[2] = {2,2};
  size_t zs = 2;
  NNLib::Tensor<int> zero = NNLib::Zeros<int>(za, zs);
  // std::cout << "zero matrix size: " << zero.matrix_size() << std::endl;

  NNLib::Tensor<int> zero_like = NNLib::Zeros_like<int>(zero);
  // std::cout << "zero_like matrix size: " << zero_like.matrix_size() << std::endl;

  NNLib::Tensor<int> one = NNLib::Ones<int>(za, zs);
  // std::cout << "one matrix size: " << one.matrix_size() << std::endl;
  
  NNLib::Tensor<float> rand_int = NNLib::Random_decimal<float>(za, zs);
  // std::cout << "rand decimal matrix size: " << rand_int.matrix_size() << std::endl;
  // std::cout << "rand decimal data: " << std::endl;
  // for (size_t i = 0; i < rand_int.matrix_size(); i++) {
  //   std::cout << rand_int.data[i] << std::endl;
  // };

  NNLib::Tensor<int> eye_one = NNLib::Eye<int>(za, zs);
  // std::cout << "eye_one matrix size: " << eye_one.matrix_size() << std::endl;

  NNLib::Tensor<int> cat = NNLib::Concat<int>(zero, one, 0);
  // std::cout << "cat matrix size: " << cat.matrix_size() << std::endl;


  if (__cplusplus == 201703L) {std::cout << "C++17";}
  else if (__cplusplus == 201402L) {std::cout << "C++14";}
  else if (__cplusplus == 201103L) {std::cout << "C++11";}
  else if (__cplusplus == 199711L) {std::cout << "C++98";}
  else {std::cout << "pre-standard C++";};


  int arr1[] = {0,1,2,3,4,5};
  int arr2[] = {0,1,2,3,4,5};
  size_t arr_s_1 = 6;
  size_t arr_s_2 = 6;
  int arr_shape[2] = {2,3};
  size_t arr_shape_size = 2;
  NNLib::Tensor<int> add1(arr1, arr_s_1, arr_shape, arr_shape_size);
  NNLib::Tensor<int> add2(arr2, arr_s_2, arr_shape, arr_shape_size);
  NNLib::Tensor<int> mat_add = NNLib::mat_add<int>(add1, add2);
  std::cout << "add matrix size: " << mat_add.matrix_size() << std::endl;

  // int ta[4] = {2,3,4,5};
  // size_t ts = 4;
  // NNLib::Tensor<float> rand_f = NNLib::Random_decimal<float>(ta, ts);
  // NNLib::Tensor<float> transpose = NNLib::Transpose<float>(rand_f, 1, 2);
  // std::cout << "transpose matrix size: " << transpose.matrix_size() << std::endl;

  if (__cplusplus == 201703L) {std::cout << "C++17";}
  else if (__cplusplus == 201402L) {std::cout << "C++14";}
  else if (__cplusplus == 201103L) {std::cout << "C++11";}
  else if (__cplusplus == 199711L) {std::cout << "C++98";}
  else {std::cout << "pre-standard C++";};

  
  return 0;
}



// g++ main.cpp -o main.exe MathLib.cpp