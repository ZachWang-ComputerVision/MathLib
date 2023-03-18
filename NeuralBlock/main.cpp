#include <iostream>
#include <vector>
#include <list>
#include <cstdlib>
#include <omp.h>

#include <fstream>
#include <sstream>
// #include <tbb/detail/

#include "json.h"
// #include "NNLib.h"
// #include "test.cuh"



// template <typename T> NNLib::Tensor<T> AConcat(NNLib::Tensor<T> tensor1, NNLib::Tensor<T> tensor2, int dim) {
//   int* shape1 = tensor1.matrix_shape();
//   int* shape2 = tensor2.matrix_shape();
//   if (dim < 0 || dim > (int)tensor1.shape_size() - 1) { throw std::invalid_argument("Dimension is out of range."); };

//   T* data1 = tensor1.get_data();
//   T* data2 = tensor2.get_data();

//   // this must be out size, not the entire vector size; otherwise
//   int size = 1;
//   size_t new_shape_size = tensor1.shape_size();

//   static int* new_shape = new int[(int)new_shape_size];

//   for (int i = 0; i < (int)tensor1.shape_size(); i++) {
//     if (i != dim) {
//       if (shape1[i] != shape2[i]) { throw std::invalid_argument("The dimensions of two matrices are not compitable."); };
//       new_shape[i] = shape1[i];      
//     } 
//     else {
//       new_shape[i] = shape1[i] + shape2[i];
//     };
//     // get the size for the first matrix
//     size *= shape1[i];
//   };

//   // chunk indicates how many item in a chunk. Since I am working with concatenation, it means that
//   // I combine one chunk from matrix 1 and one chunk from matrix 2 to become one chunk of the new matrix. 
//   // Then, the process repeats until the loop goes over the entire matrices.
//   int chunk1 = 1;
//   int chunk2 = 1;
//   for (int i = dim; i < (int)tensor1.shape_size(); i++) {
//     chunk1 *= shape1[i];
//     chunk2 *= shape2[i];
//   };

//   int new_size = 1;
//   for (int i = 0; i < (int)new_shape_size; i++) { new_size *= new_shape[i]; };

//   static T* new_vec = new T[new_size];

//   int new_vec_idx = 0;
//   // The outer loop goes through n chunks. The inner loop runs within the chunk to fill the new matrix.
//   for (int i = 0; i <= (size / chunk1 - 1); i++) {
//     for (int j = 0; j < chunk1; j++) {
//       int idx = j + i * chunk1;
//       new_vec[new_vec_idx] = data1[idx];
//       new_vec_idx++;
//     };
//     for (int k = 0; k < chunk2; k++) {
//       int idx = k + i * chunk2;
//       new_vec[new_vec_idx] = data2[idx];
//       new_vec_idx++;
//     };
//   };

//   NNLib::Tensor<T> new_tensor(new_vec, (size_t)new_size, new_shape, new_shape_size);
//   return new_tensor;
// };



int main() {
    /* Test JSON type container */
    std::cout << "Test JSON type container" << std::endl;
    NNLib::JSON v1;
    NNLib::JSON v2 = true;
    NNLib::JSON v3 = 123;
    NNLib::JSON v4 = 1.23;
    NNLib::JSON v5 = "hello";

    NNLib::JSON arr;
    arr.append(true);
    arr.append(123);
    arr.append(1.23);
    arr.append("hello");

    std::cout << arr.str() << std::endl;
    std::cout << "use iterator: ";
    for (auto item = arr.begin(); item != arr.end(); item++) {
        std::cout << (*item).str() << ", ";
    };
    std::cout << "" << std::endl;
    std::cout << " " << std::endl;

    bool b = arr[0];
    int i = arr[1];
    double f = arr[2];
    const std::string& s = arr[3];


    NNLib::JSON obj;
    obj["bool"] = true;
    obj["int"] = 123;
    obj["float"] = 1.23;
    obj["hello"] = "Hello there";

    std::cout << obj.str() << std::endl;

    bool hasint = obj.has(v3);
    bool hasstr = obj.has(v5);
    std::cout << "hasint: " << hasint << ", " << "hasstr: " << hasstr << std::endl;

    obj.remove(v5);
    bool recheck_hasstr = obj.has(v5);
    std::cout << "recheck_hasstr: " << recheck_hasstr << std::endl;

    arr.clear();
    obj.clear();

    std::cout << "---------------------------------------------" << std::endl;

    // JSON parser
    // const std::string & str_parser = "\"hello world\"";
    // const std::string & str_parser = "[1, 2, 3]";
    const std::string& str_parser = "{\"a\": 1, \"b\": 2}";
    NNLib::Parser q;
    q.load(str_parser);
    NNLib::JSON p = q.parse();
    std::cout << p.str() << std::endl;

    // read a json file
    // ifstream fin("./test.json");
    // stringstream ss;
    // const std::string & f_content = ss.str();
    // NNLib::Parser q;
    // q.load(str_parser);
    // NNLib::JSON p = q.parse();
    // std::cout << p.str() << std::endl;




    /* Test Image and Video read */



    // 
    // const int tre = 5;
    // int trew[tre];
    // trew[0] = 1;
    // trew[1] = 1;
    // trew[2] = 1;
    // trew[3] = 1;
    // std::cout << "out: " << trew[0] << trew[1] << trew[2] << std::endl;

    // int za[2] = {2,3};
    // size_t zs = 2;

    // int a[6] = {1,2,3,4,5,6};
    // int b[6] = {1,2,3,4,5,6};
    // size_t s = 6;

    // NNLib::Tensor<int> zero = NNLib::Tensor<int>(a, s, za, zs);
    // NNLib::Tensor<int> one = NNLib::Tensor<int>(a, s, za, zs);

    // NNLib::Tensor<int> acon = AConcat<int>(zero, one, 0);
    // int* dataa = acon.get_data();
    // std::cout << dataa[0] << std::endl;
    // std::cout << dataa[1] << std::endl;
    // std::cout << dataa[2] << std::endl;
    // std::cout << dataa[3] << std::endl;
    // std::cout << dataa[2] << std::endl;
    // std::cout << dataa[2] << std::endl;
    // std::cout << dataa[2] << std::endl;

    // delete[] acon.data;
    // delete[] acon.dims;

    // NNLib::Tensor<int> zero = NNLib::Zeros<int>(za, zs);
    // std::cout << "zero matrix size: " << zero.matrix_size() << std::endl;

    // NNLib::Tensor<int> zero_like = NNLib::Zeros_like<int>(zero);
    // std::cout << "zero_like matrix size: " << zero_like.matrix_size() << std::endl;

    // NNLib::Tensor<int> one = NNLib::Ones<int>(za, zs);
    // std::cout << "one matrix size: " << one.matrix_size() << std::endl;

    // NNLib::Tensor<float> rand_int = NNLib::Random_decimal<float>(za, zs);
    // std::cout << "rand decimal matrix size: " << rand_int.matrix_size() << std::endl;

    // NNLib::Tensor<int> eye_one = NNLib::Eye<int>(za, zs);
    // std::cout << "eye_one matrix size: " << eye_one.matrix_size() << std::endl;


    // int* zeroarray = zero.get_data();
    // int* onearray = one.get_data();
    // std::cout << "start: " << std::endl;
    // std::cout << zeroarray[0] << std::endl;
    // std::cout << zeroarray[1] << std::endl;
    // std::cout << zeroarray[2] << std::endl;
    // std::cout << zeroarray[3] << ", " << onearray[3] << std::endl;


    // NNLib::Tensor<int> cat = AConcat<int>(zero, one, 0);
    // int* cat_data = cat.get_data();
    // std::cout << "cat matrix size: " << cat.matrix_size() << std::endl;
    // std::cout << "cat matrix data: ";
    // for (int i = 0; i < 8; i ++) {
    //   std::cout << cat_data[i] << ", ";
    // };
    // std::cout << "end" << std::endl;



    // int arr1[] = {0,1,2,3,4,5};
    // int arr2[] = {0,1,2,3,4,5};
    // size_t arr_s_1 = 6;
    // size_t arr_s_2 = 6;
    // int arr_shape[2] = {2,3};
    // size_t arr_shape_size = 2;
    // NNLib::Tensor<int> add1(arr1, arr_s_1, arr_shape, arr_shape_size);
    // NNLib::Tensor<int> add2(arr2, arr_s_2, arr_shape, arr_shape_size);
    // NNLib::Tensor<int> mat_add = NNLib::mat_add<int>(add1, add2);
    // int* data = mat_add.data;
    // std::cout << "add matrix size: " << mat_add.matrix_size() << std::endl;
    // for (int i = 0; i < 6; i ++) { 
    //   std::cout << data[i] << std::endl;
    // };

    // int ta[4] = {2,3,4,5};
    // size_t ts = 4;
    // NNLib::Tensor<float> rand_f = NNLib::Random_decimal<float>(ta, ts);
    // NNLib::Tensor<float> transpose = NNLib::Transpose<float>(rand_f, 1, 2);
    // std::cout << "transpose matrix size: " << transpose.matrix_size() << std::endl;

    if (__cplusplus == 201703L) { std::cout << "C++17"; }
    else if (__cplusplus == 201402L) { std::cout << "C++14"; }
    else if (__cplusplus == 201103L) { std::cout << "C++11"; }
    else if (__cplusplus == 199711L) { std::cout << "C++98"; }
    else { std::cout << "pre-standard C++"; };


    return 0;
}

