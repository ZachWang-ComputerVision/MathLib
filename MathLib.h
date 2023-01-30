#include <vector>

#ifndef MATHLIB_H
#define MATHLIB_H

namespace MathLib {
  
  template <class T> class Mat {
    private:
      std::vector<int> dims;
    public:
      T data;
      Mat(T& matrix, std::vector<int>& matShape) {
        if (matShape.size() != 4) {
          throw std::invalid_argument("Must specify the matrix's shape with four dimensions");
        };
        data = matrix;
        dims = matShape;
      };

      std::vector<int> shape() { return dims; };

      void reshape(std::vector<int>& newShape) {
        int curSize = 1;
        int newSize = 1;
        if (newShape.size() != 4) {
          throw std::invalid_argument("The new shape format must be in four dimensions");
        };

        for (int i = 0; i < 4; i++) {
          curSize *= dims[i];
        };
        for (int i = 0; i < 4; i++) {
          newSize *= newShape[i];
        };

        if (curSize == newSize) {
          dims = newShape;
        } else {
          throw std::invalid_argument("The current shape cannot be converted to the new shape");
        };
      };

      void print() {};
      ~Mat() {};
  };


  template <typename T> Mat<std::vector<T>> zeros(std::vector<int>& shape) {
    if (shape.size() != 4) {
      throw std::invalid_argument("The new shape format must be in four dimensions");
    };
    int size = 1;
    for (int i = 0; i < 4; i++) { size *= shape[i]; };
    std::vector<T> vec(size, 0.0);
    Mat<std::vector<T>> mat(vec, shape);
    return mat;
  };


  template <typename T> Mat<std::vector<T>> ones(std::vector<int>& shape) {
    if (shape.size() != 4) {
      throw std::invalid_argument("The new shape format must be in four dimensions");
    };
    int size = 1;
    for (int i = 0; i < 4; i++) { size *= shape[i]; };
    std::vector<T> vec(size, 1.0);
    Mat<std::vector<T>> mat(vec, shape);
    return mat;
  };


  template <typename T> Mat<std::vector<T>> random(std::vector<int>& shape) {
    if (shape.size() != 4) {
      throw std::invalid_argument("The new shape format must be in four dimensions");
    };
    int size = 1;
    for (int i = 0; i < 4; i++) { size *= shape[i]; };
    std::vector<T> vec;
    vec.reserve(size);
    for (int i = 0; i < size; i++) { vec.push_back( (T)(rand() % 100) ); };
    Mat<std::vector<T>> mat(vec, shape);
    return mat;
  };


  Mat<std::vector<bool>> boolean(bool state, std::vector<int>& shape) {
    if (shape.size() != 4) {
      throw std::invalid_argument("The new shape format must be in four dimensions");
    };
    int size = 1;
    for (int i = 0; i < 4; i++) { size *= shape[i]; };
    std::vector<bool> vec(size, state);
    Mat<std::vector<bool>> mat(vec, shape);
    return mat;
  };


  template <typename T> Mat<std::vector<T>> eye(std::vector<int>& shape) {
    if (shape.size() != 4) {
      throw std::invalid_argument("The new shape format must be in four dimensions.");
    };
    if (shape[2] != shape[3]) { 
      throw std::invalid_argument("The 3rd and 4th dimensions must be the same size."); 
    };
    int size = 1;
    for (int i = 0; i < 4; i++) { size *= shape[i]; };
    std::vector<T> vec(size, 0.0);
    for (int b = 0; b < shape[0]; b++) { 
      for (int c = 0; c < shape[1]; c++) {
        for (int i = 0; i < shape[2]; i++) {
          int idx = b * shape[1] + c * shape[2] + i * shape[3] + i;
          vec[idx] = (T)1.0;
        };
      };
    };
    Mat<std::vector<T>> mat(vec, shape);
    return mat;
  };


  template <typename T> Mat<std::vector<T>> concat(std::vector<T>& matrix1, std::vector<T>& matrix2, int& dim) {
    
    return matrix1;
  };


  template <typename T> std::vector<T> transpose(std::vector<T>& matrix, std::vector<int>& curShape, int& idxA, int& idxB) { 
    if (curShape[idxA] == 1 || curShape[idxB] == 1) {
      return matrix;
    }
    else {
      // for (int b = 0; b < curShape[0]; b++) {
      //   for (int c = 0; c < curShape[1]; c++) {
      //     for (int r = 0; r < curShape[2]; r++) {
      //       for (int l = 0; l < curShape[3]; l++) {

      //       }
      //     }
      //   }
      // }
      return matrix;
    }
  };

  template <typename T> std::vector<T> slice(T* matrix, std::vector<std::vector<int>> slice_idx) {

  };
};

#endif

//   template <typename T> std::vector<T> concat();
//   template <typename T> std::vector<T> transpose();
//   template <typename T> std::vector<T> permute();

//   template <typename T> std::vector<T> vec_mul(std::vector<T>, std::vector<T>);
//   template <typename T> std::vector<T> vec_dot(std::vector<T>, std::vector<T>);
//   template <typename T> std::vector<T> vec_cro(std::vector<T>, std::vector<T>);

//   template <typename T> std::vector<T> mat_mul(std::vector<T>, std::vector<T>);
//   template <typename T> std::vector<T> mat_dot(std::vector<T>, std::vector<T>);
//   template <typename T> std::vector<T> mat_cro(std::vector<T>, std::vector<T>);

//   template <typename T> std::vector<T> Linear();
//   template <typename T> std::vector<T> MLinear();
//   template <typename T> std::vector<T> Conv2D();
//   template <typename T> std::vector<T> ModuleList();

//   template <typename T> std::vector<T> BatchNorm2D();
//   template <typename T> std::vector<T> LayerNorm();
//   template <typename T> std::vector<T> Sigmoid();
//   template <typename T> std::vector<T> ReLU();

//   template <typename T> std::vector<T> SGD();
//   template <typename T> std::vector<T> Adam();

//   template <typename T> std::vector<T> save();
//   template <typename T> std::vector<T> load();
// }