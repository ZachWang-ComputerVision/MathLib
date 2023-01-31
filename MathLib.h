#include <iostream>
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
        for (int i = 0; i < 4; i++) {
          if (matShape[i] < 1) {
            throw std::invalid_argument("All dimensional axis must be above 1, such as {1, 1, 2, 2}");
          };
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


  template <typename T> Mat<std::vector<T>> slice(Mat<std::vector<T>>& matrix, std::vector<std::vector<int>> slice_idx) {
    
    Mat<std::vector<T>> mat(matrix, shape);
    return mat;
  };

  std::vector<float> slice(std::vector<float> vector, int start, int end) {
    std::vector<float> subvec;
    return subvec;
  };


  template <typename T> Mat<std::vector<T>> concat(Mat<std::vector<T>>& matrix1, std::vector<T>& matrix2, int& dim) {
  
    Mat<std::vector<T>> mat(matrix1, shape);
    return mat;
  };


  template <typename T> Mat<std::vector<T>> transpose(Mat<std::vector<T>>& matrix, int idxA, int idxB) { 
    
    
    if (curShape[idxA] == 1 || curShape[idxB] == 1) {
      return matrix;
    }
    else {
      
      Mat<std::vector<T>> mat(matrix, shape);
      return mat;
    }
  };


  template <typename T> Mat<std::vector<T>> permute(Mat<std::vector<T>>& matrix, int& idxA, int& idxB) {
    
    Mat<std::vector<T>> mat(matrix, shape);
    return mat;
  };


  Mat<std::vector<float>> mat_add(Mat<std::vector<float>>& mat1, Mat<std::vector<float>>& mat2) {
    std::vector<int> m1Shape = mat1.shape();
    std::vector<int> m2Shape = mat2.shape();

    int range = 1;
    for (int i = 0; i < 4; i++) {
      if (m1Shape[i] < m2Shape[i]) {
        throw std::invalid_argument("One of the axis of Matrix 1 has a smaller value than the axis of Matrix 2");
      };
      if (m1Shape[i] != m2Shape[i] && m2Shape[i] != 1) {
        throw std::invalid_argument("The value of Matrix 2's axis must be either 1 or the same to the Matrix 1's");
      };
      if (m1Shape[1] != m2Shape[1]) {
        int multi = m1Shape[1] / m2Shape[1];
        range *= multi;
      };
    };
    
    int mat1Length = 1;
    for (int dim : m1Shape) { mat1Length *= dim; };

    std::vector<float> data1 = mat1.data;
    std::vector<float> data2 = mat2.data;
    std::vector<float> newVector;
    newVector.reserve(mat1Length);

    int mat2Length = 1;
    for (int dim : m2Shape) { mat2Length *= dim; };

    for (int l = 0; l < mat2Length; l++) {
      for (int r = 0; r < range; r++) {
        int idx = l + r * mat2Length;
        newVector[idx] = data1[idx] + data2[l];
      }; 
    };

    Mat<std::vector<float>> mat(data1, mat1.shape());
    return mat;
  };


  Mat<std::vector<float>> mat_mul(Mat<std::vector<float>>& mat1, Mat<std::vector<float>>& mat2) {
    std::vector<int> m1Shape = mat1.shape();
    std::vector<int> m2Shape = mat2.shape();

    int range = 1;
    for (int i = 0; i < 4; i++) {
      if (m1Shape[i] < m2Shape[i]) {
        throw std::invalid_argument("One of the axis of Matrix 1 has a smaller value than the axis of Matrix 2");
      };
      if (m1Shape[i] != m2Shape[i] && m2Shape[i] != 1) {
        throw std::invalid_argument("The value of Matrix 2's axis must be either 1 or the same to the Matrix 1's");
      };
      if (m1Shape[1] != m2Shape[1]) {
        int multi = m1Shape[1] / m2Shape[1];
        range *= multi;
      };
    };
    
    int mat1Length = 1;
    for (int dim : m1Shape) { mat1Length *= dim; };

    std::vector<float> data1 = mat1.data;
    std::vector<float> data2 = mat2.data;
    std::vector<float> newVector;
    newVector.reserve(mat1Length);

    int mat2Length = 1;
    for (int dim : m2Shape) { mat2Length *= dim; };

    for (int l = 0; l < mat2Length; l++) {
      for (int r = 0; r < range; r++) {
        int idx = l + r * mat2Length;
        newVector[idx] = data1[idx] * data2[l];
      }; 
    };

    Mat<std::vector<float>> mat(data1, mat1.shape());
    return mat;
  };


  Mat<std::vector<float>> mat_dot(Mat<std::vector<float>>& mat1, Mat<std::vector<float>>& mat2) {
    std::vector<int> m1Shape = mat1.shape();
    std::vector<int> m2Shape = mat2.shape();

    if (m2Shape[2] != m1Shape[3] || m2Shape[0] != 1 || m2Shape[1] != 1) {
      throw std::invalid_argument("Dimensions are incompatible for dot product.");
    };
    std::vector<int> newShape = {m1Shape[0], m1Shape[1], m1Shape[2], m2Shape[3]};
    
    int range = (m1Shape[0] / m2Shape[0]) * (m1Shape[1] / m2Shape[1]);
    
    int mat1Length = 1;
    for (int dim : m1Shape) { mat1Length *= dim; };
    int newLength = 1;
    for (int dim : newShape) { newLength *= dim; };
    int mat2Length = 1;
    for (int dim : m2Shape) { mat2Length *= dim; };

    std::vector<float> data1 = mat1.data;
    Mat<std::vector<float>> transData2 = transpose<float>(mat2, 2, 3);
    std::vector<float> data2 = transData2.data;

    std::vector<float> newVector;
    newVector.reserve(newLength);

    for (int i = 0; i < mat1Length / m1Shape[3]; i++) {
      for (int j = 0; j < mat2Length / m2Shape[2]; j++) {
        float subsum = 0;
        for (int k = 0; k < m1Shape[3]; k++) {
          int idx1 = k + i * m1Shape[3];
          int idx2 = k + j * m2Shape[2];
          subsum += (data1[idx1] * data2[idx2]);
        };
        newVector[j + i * m1Shape[3]] = subsum;
      }; 
    };

    Mat<std::vector<float>> newMat(newVector, newShape);
    return newMat;
  };

  

};

#endif

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