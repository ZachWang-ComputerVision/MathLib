#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "Tensor.h"

#ifndef NNLIB_H
#define NNLIB_H

namespace NNLib {

  template <typename T> Tensor<std::vector<T>> normalize_imgs(Tensor<std::vector<T>>& mat, std::vector<int>& shift_values) {
    
  };

  template <typename T> Tensor<std::vector<T>> zeros(std::vector<int>& shape) {
    int size = 1;
    for (int i = 0; i < shape.size(); i++) { size *= shape[i]; };
    std::vector<T> vec(size, 0.0);
    Tensor<std::vector<T>> mat(vec, shape);
    return mat;
  };


  template <typename T> Tensor<std::vector<T>> ones(std::vector<int>& shape) {
    int size = 1;
    for (int i = 0; i < shape.size(); i++) { size *= shape[i]; };
    std::vector<T> vec(size, 1.0);
    Tensor<std::vector<T>> mat(vec, shape);
    return mat;
  };


  Tensor<std::vector<int>> randomInt(std::vector<int>& shape) {
    int size = 1;
    for (int i = 0; i < shape.size(); i++) { size *= shape[i]; };
    std::vector<int> vec;
    vec.reserve(size);
    for (int i = 0; i < size; i++) { vec[i] = (int)(rand() % 100); };
    Tensor<std::vector<int>> mat(vec, shape);
    return mat;
  };


  template <typename T> Tensor<std::vector<T>> randomDecimal(std::vector<int>& shape) {
    int size = 1;
    for (int i = 0; i < shape.size(); i++) { size *= shape[i]; };
    std::vector<T> vec;
    vec.reserve(size);
    for (int i = 0; i < size; i++) { vec[i] = (T)(rand() % 100) / 100.0; };
    Tensor<std::vector<T>> mat(vec, shape);
    return mat;
  };


  Tensor<std::vector<bool>> boolean(bool state, std::vector<int>& shape) {
    int size = 1;
    for (int i = 0; i < shape.size(); i++) { size *= shape[i]; };
    std::vector<bool> vec(size, state);
    Tensor<std::vector<bool>> mat(vec, shape);
    return mat;
  };


  template <typename T> Tensor<std::vector<T>> eye(std::vector<int>& shape) {
    int last2nd = shape[shape.size() - 2];
    int last1st = shape[shape.size() - 1];
    if (last2nd != last1st) { 
      throw std::invalid_argument("The inside matrix is not a square"); 
    };
    int size = 1;
    for (int i = 0; i < shape.size; i++) { size *= shape[i]; };
    
    std::vector<T> vec(size, 0.0);

    for (int i = 0; i < size / (last2nd * last1st); i++) {
      for (int j = 0; j < last1st; j++) {
        int idx = j + j * last1st + i * last2nd * last1st;
        vec[idx] = (T)1.0;
      };
    };

    Tensor<std::vector<T>> mat(vec, shape);
    return mat;
  };


  // template <typename T> Mat<std::vector<T>> slice(Mat<std::vector<T>>& mat, std::vector<int>& indices) {
  //   std::vector<T> data = mat.data;
  //   std::vector<int> shape = mat.shape();
  //   std::vector<int> newShape;

  //   // pos contains all the starting indices and keeps tracking if there is any dimension is 

  //   int newSize = 1;
  //   for (int i = 0; i < indices.size(); i++) {
  //     if (indices[i].size() != 2) { throw std::invalid_argument("One of axis's slicing indices is in a wrong form"); };
  //     int s = indices[i][0];
  //     int e = indices[i][1];
  //     if (e < s) { throw std::invalid_argument("Begining or end index is set wrong."); };
  //     if (s < 0 || e > shape[i]) { throw std::invalid_argument("Index is out of range."); };
  //     int dimVal = e - s + 1;
  //     newSize *= dimVal;
  //     newShape.push_back(dimVal);
  //   };

  //   std::vector<T> subVec;
  //   subVec.reserve(newSize);

  //   bool end = false;
  //   int count = 0;
  //   int chunk = newShape.back();
  //   // int idx = pos.back();
  //   while (end != true) {
  //     // Going through the list in backwards to calculate the index.
  //     for (int i = newShape.size() - 2; i > -1; i--) {
        
  //       // Get position and push the items into the subVec
  //       for (int k = shape.size() - 2; k > -1; k--) {
  //         idx += pos[k] * chunk;
  //         newMat[count] = data[idx];
  //         count++;
  //         pos[i]++;
  //       };

  //       // Check if the axis's end is reached or not. If it does, update the pos track
  //       if (pos[i] == indices[i][1]) {
  //         if (i > 0) {
  //           pos[i - 1]++;
            
  //         }

  //         for (int k = i; k < pose.size(); k++) {
  //           pos[i] = indices[i][0];
  //         };

  //         // All the axis are upda
  //         else {
  //           // check if all axis meet the end
  //           int count_req = 0;
  //           for (int n = 1; n < pose.size(); n++) {
  //             if (pos[n] == indices[i][1]) {
  //               count_req++;
  //             };
  //           };
  //           if ()
  //           end = true;
  //         }

  //       };
  //     };
  //   };
    
  //   Mat<std::vector<T>> subMat(subVec, newSize);
  //   return subMat;
  // };


  template <typename T> Tensor<std::vector<T>> concat(Tensor<std::vector<T>>& matrix1, std::vector<T>& matrix2, int& dim) {
    std::vector<int> shape1 = matrix1.shape();
    std::vector<int> shape2 = matrix2.shape();
    if (dim < 0 || dim > shape1.size() - 1) { throw std::invalid_argument("Dimension is out of range."); };

    std::vector<T> data1 = matrix1.data;
    std::vector<T> data2 = matrix2.data;

    int size = 1;
    std::vector<int> newShape(shape1, 1);

    for (int i = 0; i < shape1; i++) {
      if (i != dim) {
        if (shape1[i] != shape2[i]) { throw std::invalid_argument("The dimensions of two matrices are not compitable."); };
        newShape[i] = shape1[i];
      } 
      else {
        newShape[i] = shape1[i] + shape2[i];
      };
      size *= shape1[i];
    };

    // chunk indicates how many item in a chunk. Since I am working with concatenation, it means that
    // I combine one chunk from matrix 1 and one chunk from matrix 2 to become one chunk of the new matrix. 
    // Then, the process repeats until the loop goes over the entire matrices.
    int chunk1 = 1;
    int chunk2 = 1;
    for (int i = dim; i < shape1.size(); i++) {
      chunk1 *= shape1[i];
      chunk2 *= shape2[i];
    };

    int newSize = 1;
    for (int i = 0; i < newShape.size(); i++) { newSize *= newShape[i]; };

    std::vector<T> newMat;
    newMat.reserve(newSize);

    int newMatIdx = 0;
    // The outer loop goes through n chunks. The inner loop runs within the chunk to fill the new matrix.
    for (int i = 0; i < size / chunk1; i++) {
      for (int j = 0; j < chunk1; j++) {
        int idx = j + i * chunk1;
        newMat[newMatIdx] = data1[idx];
        newMatIdx++;
      };
      for (int k = 0; k < chunk2; k++) {
        int idx = k + i * chunk2;
        newMat[newMatIdx] = data2[idx];
        newMatIdx++;
      };
    };
    Tensor<std::vector<T>> mat(newMat, newShape);
    return mat;
  };


  template <typename T> Tensor<std::vector<T>> transpose(Tensor<std::vector<T>>& matrix, int idxA, int idxB) { 
    std::vector<int> shape = matrix.shape();
    std::vector<T> data = matrix.data;
    
    if (idxA < 0 || idxA > shape.size() - 1 || idxB < 0 || idxB > shape.size() - 1) {
      throw std::invalid_argument("Index is out of range.");
    }
    if (idxB - idxA != 1) { throw std::invalid_argument("Transpose must be applied to next dimension."); };
    
    int target = shape[idxB];
    shape[idxB] = shape[idxA];
    shape[idxA] = target;
    
    // The outer loop calculates how many loops we need to go through before the two transpose indices
    // The inner loop calculates how many loops we need to go through after the two transpose indices
    // 2 x 2 x 5 x 3 x 2 x 2  if we transpose 5 and 3 , the outer loop is 4 = 2 x 2, and the inner loop is 4 = 2 x 2
    int outerLoop = 1;
    int innerLoop = 1;
    for (int i = 0; i < idxA; i ++) { outerLoop *= shape[i]; };
    for (int i = idxB + 1; i < shape.size(); i++) { innerLoop *= shape[i]; };
    
    // The outerChunk is 5 x 3 x 2 x 2 if we use the above example.
    // The innerChunk is 2 x 2, which is same to the innerLoop.
    int outerChunk = 1;
    int innerChunk = innerLoop;
    for (int i = idxA; i < shape.size(); i++) {
      outerChunk *= shape[i];
    };

    std::vector<T> newMatrix;
    newMatrix.reserve(outerLoop * outerChunk);

    int count = 0;
    for (int i = 0; i < outerLoop; i++) {
      for (int j = 0; j < shape[idxA]; j++) {
        for (int k = 0; k < shape[idxB]; k++) {
          int h = k + j * shape[idxB] + i * outerChunk;
          for (int w = 0; w < innerLoop; w++) {
            int idx = w + h * innerChunk;
            newMatrix[count] = data[idx];
            count++;
          };
        };
      };
    };

    Tensor<std::vector<T>> mat(newMatrix, shape);
    return mat;
  };


  template <typename T> Tensor<std::vector<T>> permute(Tensor<std::vector<T>>& matrix, int& idxA, int& idxB) {
    std::vector<int> shape = matrix.shape();
    Tensor<std::vector<T>> mat(matrix, shape);
    return mat;
  };


  // Implement a function for arithmatic calculation

  template <typename T> void perform_operation(T (*opera)(T, T), std::vector<T>& newVector, Tensor<std::vector<T>>& m1, Tensor<std::vector<T>>& m2, int dimPrt, int& idx, int s1, int e1, int s2, int e2) {
    std::vector<int> shape1 = m1.shape();
    std::vector<int> shape2 = m2.shape();

    if (dimPrt == shape1.size() - 1) {
      if (shape1[dimPrt] == shape2[dimPrt]) {
        for (i = 0; i < e1 - s1 + 1; i ++) {
          newVector[idx] = opera(m1.data[i + s1], m2.data[i + s2]);  // Implement a function for arithmatic calculation
          idx ++;
        };
      };
      else {
        for (int i = s1; i < e1; i ++) {
          for (int j = s2; j < e2; j ++) {
            newVector[idx] = opera(m1.data[i + s1], m2.data[i + s2]);  // Implement a function for arithmatic calculation
            idx ++;
          };
        };
      };
    }
    else {
      int chunk1 = (e1 - s1 + 1) / shape1[dimPrt];
      int chunk2 = (e2 - s2 + 1) / shape2[dimPrt];
      int nextDimPrt = dimPrt + 1;
      if (shape1[dimPrt] == shape2[dimPrt]) {
        for (int i = 0; i < shape1[dimPrt]; i ++) {
          int newS1 = s1 + i * chunk1;
          int newE1 = e1 + i * chunk1;
          int newS2 = s2 + i * chunk2;
          int newE2 = e2 + i * chunk2;
          perform_operation<T>(opera, newVector, m1, m2, nextDimPrt, idx, newS1, newE1, newS2, newE2);
        };
      }
      else {
        for (int i = 0; i < shape1[dimPrt], i ++) {
          int newS1 = s1 + i * chunk1;
          int newE1 = e1 + i * chunk1;
          for (int j = 0; j < shape2[dimPrt], j ++) {
            int newS2 = s2 + i * chunk2;
            int newE2 = e2 + i * chunk2;
            perform_operation<T>(opera, newVector, m1, m2, nextDimPrt, idx, newS1, newE1, newS2, newE2);
          };
        };
      };
    };
  };

  void update_shape_n_size(std::vector<int>& m1Shape, std::vector<int>& m2Shape, std::vector<int>& newShape, int& newSize) {
    if (m1Shape.size() != m2Shape.size()) { 
      throw std::invalid_argument("The number of dimension is different between the two matrices."); 
    };
    if (m1Shape.size() < 1) {
      throw std::invalid_argument("The matrix cannot be empty."); 
    };
    
    for (int i = 0; i < m1Shape.size(); i++) {
      if (m1Shape[i] != m2Shape[i]) {
        if (m1Shape[i] != 1 && m2Shape[i] != 1) {
          throw std::invalid_argument("The dimensions are incompatible.");
        };
      };

      int n = std::max(m1Shape[i], m2Shape[i]);
      newShape[i] = n;
      newSize *= n;
    };
  };


  template <typename T> Tensor<std::vector<T>> mat_add(Tensor<std::vector<T>>& mat1, Tensor<std::vector<T>>& mat2) {
    std::vector<int> m1Shape = mat1.shape();
    std::vector<int> m2Shape = mat2.shape();
    std::vector<int> newShape(m1Shape.size(), 1);
    int newSize = 1;
    update_shape_n_size(m1Shape, m2Shape, newShape, newSize);

    std::vector<T> data1 = mat1.data;
    std::vector<T> data2 = mat2.data;
    std::vector<T> newVector;
    newVector.reserve(newSize);

    perform_operation<T>([](T a, T b){ return a + b }, newVector, mat1, mat2, 0, 0, 0, size1, 0, size2);
    Tensor<std::vector<T>> mat(newVector, newShape);
    return mat;
  };


  template <typename T> Tensor<std::vector<T>> mat_sub(Tensor<std::vector<T>>& mat1, Tensor<std::vector<T>>& mat2) {
    std::vector<int> m1Shape = mat1.shape();
    std::vector<int> m2Shape = mat2.shape();
    std::vector<int> newShape(m1Shape.size(), 1);
    int newSize = 1;
    update_shape_n_size(m1Shape, m2Shape, newShape, newSize);

    std::vector<T> data1 = mat1.data;
    std::vector<T> data2 = mat2.data;
    std::vector<T> newVector;
    newVector.reserve(newSize);

    perform_operation<T>([](T a, T b){ return a - b }, newVector, mat1, mat2, 0, 0, 0, size1, 0, size2);
    Tensor<std::vector<T>> mat(newVector, newShape);
    return mat;
  };


  template <typename T> Tensor<std::vector<T>> mat_mul(Tensor<std::vector<T>>& mat1, Tensor<std::vector<T>>& mat2) {
    std::vector<int> m1Shape = mat1.shape();
    std::vector<int> m2Shape = mat2.shape();
    std::vector<int> newShape(m1Shape.size(), 1);
    int newSize = 1;
    update_shape_n_size(m1Shape, m2Shape, newShape, newSize);

    std::vector<T> data1 = mat1.data;
    std::vector<T> data2 = mat2.data;
    std::vector<T> newVector;
    newVector.reserve(newSize);

    perform_operation<T>([](T a, T b){ return a * b }, newVector, mat1, mat2, 0, 0, 0, size1, 0, size2);
    Tensor<std::vector<T>> mat(newVector, newShape);
    return mat;
  };


  template <typename T> Tensor<std::vector<T>> mat_div(Tensor<std::vector<T>>& mat1, Tensor<std::vector<T>>& mat2) {
    std::vector<int> m1Shape = mat1.shape();
    std::vector<int> m2Shape = mat2.shape();
    std::vector<int> newShape(m1Shape.size(), 1);
    int newSize = 1;
    update_shape_n_size(m1Shape, m2Shape, newShape, newSize);

    std::vector<T> data1 = mat1.data;
    std::vector<T> data2 = mat2.data;
    std::vector<T> newVector;
    newVector.reserve(newSize);

    perform_operation<T>([](T a, T b){ return a / b }, newVector, mat1, mat2, 0, 0, 0, size1, 0, size2);
    Tensor<std::vector<T>> mat(newVector, newShape);
    return mat;
  };


  template <typename T> Tensor<std::vector<T>> mat_dot(Tensor<std::vector<T>>& mat1, Tensor<std::vector<T>>& mat2) {
    
    Tensor<std::vector<T>> newMat(mat1, mat1.shape());
    return newMat;
  };

  
  template <typename T> Tensor<std::vector<T>> Conv2d(Tensor<std::vector<T>>& mat, int in_channels, int out_channels, int kernel, int padding, int stride) {
    Tensor<std::vector<T>> newMat(mat, mat.shape());
    return newMat;
  };

  template <typename T> Tensor<std::vector<T>> BatchNorm2D(Tensor<std::vector<T>>& mat) {
    /*
    num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None
    
    num_features (int) – CC from an expected input of size (N, C, H, W)(N,C,H,W)
    eps (float) – a value added to the denominator for numerical stability. Default: 1e-5
    momentum (float) – the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
    affine (bool) – a boolean value that when set to True, this module has learnable affine parameters. Default: True
    track_running_stats (bool) – a boolean value that when set to True, this module tracks the running mean and variance, and when set to False, this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None. When these buffers are None, this module always uses batch statistics. in both training and eval modes. Default: True
    */

  };

  template <typename T> void ReLU(Tensor<std::vector<T>>& mat) {
    T zero = 0.0;
    for (auto& item : mat.data) { item = std::max(zero, item); };
  };

  template <typename T> void Sigmoid(Tensor<std::vector<T>>& mat) {
    T one = 1.0;
    for (auto& item : mat.data) { item = one / (one + std::exp(-item)); };
  };
  
};

// Adam  https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
// WightInit  https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
// AutoGrad  https://github.com/joelgrus/autograd/blob/part06/tests/test_tensor_matmul.py
//           https://github.com/joelgrus/autograd/blob/part06/autograd/tensor.py

// CUDA  https://forums.developer.nvidia.com/t/how-to-call-cuda-function-from-c-file/61986/3
//       https://forums.developer.nvidia.com/t/how-to-use-class-in-cuda-c/61761
// CUDA  From Scratch

#endif

//   template <typename T> std::vector<T> Linear();
//   template <typename T> std::vector<T> MLinear();
//   template <typename T> std::vector<T> Conv2D();


//   template <typename T> std::vector<T> LayerNorm();

//   template <typename T> std::vector<T> Adam();

//   template <typename T> std::vector<T> save();
//   template <typename T> std::vector<T> load();

//   multi-thread
//   CUDA
