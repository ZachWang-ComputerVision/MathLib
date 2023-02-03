#include <iostream>
#include <vector>
#include <cstdlib>

#ifndef MATHLIB_H
#define MATHLIB_H

namespace MathLib {
  
  template <class T> class Mat {
    private:
      std::vector<int> dims;
    public:
      T data;
      Mat(T& matrix, std::vector<int>& matShape) {
        for (int i = 0; i < matShape.size(); i++) {
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

        for (int i = 0; i < dims.size(); i++) {
          curSize *= dims[i];
        };
        for (int i = 0; i < newShape.size(); i++) {
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
    int size = 1;
    for (int i = 0; i < shape.size(); i++) { size *= shape[i]; };
    std::vector<T> vec(size, 0.0);
    Mat<std::vector<T>> mat(vec, shape);
    return mat;
  };


  template <typename T> Mat<std::vector<T>> ones(std::vector<int>& shape) {
    int size = 1;
    for (int i = 0; i < shape.size(); i++) { size *= shape[i]; };
    std::vector<T> vec(size, 1.0);
    Mat<std::vector<T>> mat(vec, shape);
    return mat;
  };


  Mat<std::vector<int>> randomInt(std::vector<int>& shape) {
    int size = 1;
    for (int i = 0; i < shape.size(); i++) { size *= shape[i]; };
    std::vector<int> vec;
    vec.reserve(size);
    for (int i = 0; i < size; i++) { vec[i] = (int)(rand() % 100); };
    Mat<std::vector<int>> mat(vec, shape);
    return mat;
  };


  Mat<std::vector<float>> randomFloat(std::vector<int>& shape) {
    int size = 1;
    for (int i = 0; i < shape.size(); i++) { size *= shape[i]; };
    std::vector<float> vec;
    vec.reserve(size);
    for (int i = 0; i < size; i++) { vec[i] = (int)(rand() % 100) / 100; };
    Mat<std::vector<float>> mat(vec, shape);
    return mat;
  };


  Mat<std::vector<bool>> boolean(bool state, std::vector<int>& shape) {
    int size = 1;
    for (int i = 0; i < shape.size(); i++) { size *= shape[i]; };
    std::vector<bool> vec(size, state);
    Mat<std::vector<bool>> mat(vec, shape);
    return mat;
  };


  template <typename T> Mat<std::vector<T>> eye(std::vector<int>& shape) {
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

    Mat<std::vector<T>> mat(vec, shape);
    return mat;
  };


  template <typename T> Mat<std::vector<T>> slice(Mat<std::vector<T>>& mat, std::vector<int>& indices) {
    
    std::vector<T> data = mat.data;
    std::vector<int> shape = mat.shape();
    std::vector<int> newShape;

    // pos contains all the starting indices and keeps tracking if there is any dimension is 
    std::vector<int> pos(shape.size(), 0);
    int newSize = 1;
    for (int i = 0; i < indices.size(); i++) {
      if (indices[i].size() != 2) { throw std::invalid_argument("One of axis's slicing indices is in a wrong form"); };
      int s = indices[i][0];
      int e = indices[i][1];
      if (e < s) { throw std::invalid_argument("Begining or end index is set wrong."); };
      if (s < 0 || e > shape[i]) { throw std::invalid_argument("Index is out of range."); };
      int dimVal = e - s + 1;
      newSize *= dimVal;
      newShape.push_back(dimVal);
      pos[i] = indices[i][0]
    };

    std::vector<T> subVec;
    subVec.reserve(newSize);

    bool end = false;
    int count = 0;
    while (end != true) {
      // Going through the list in backwards to calculate the index.
      for (int i = pos.size(); i > -1; i--) {
        
        // Get position and push the items into the subVec
        int chunk = shape.back();
        int idx = pos.back();
        for (int k = shape.size() - 2; k > -1; k--) {
          idx += pos[k] * chunk;
          newMat[count] = data[idx];
          count++;
          pos[i]++;
        };

        // Check if the axis's end is reached or not. If it does, update the pos track
        if (pos[i] == indices[i][1]) {
          if (i > 0) {
            pos[i - 1]++;
            for (int k = i; k < pose.size(); k++) {
              pos[i] = indices[i][0];
            };
          }
          // All the axis are upda
          else {
            // check if all axis meet the end
            for 
            end = true;
          }
        };
      };
    };
    
    Mat<std::vector<T>> subMat(subVec, newSize);
    return subMat;
  };


  template <typename T> Mat<std::vector<T>> concat(Mat<std::vector<T>>& matrix1, std::vector<T>& matrix2, int& dim) {
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
    Mat<std::vector<T>> mat(newMat, newShape);
    return mat;
  };


  template <typename T> Mat<std::vector<T>> transpose(Mat<std::vector<T>>& matrix, int idxA, int idxB) { 
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

    Mat<std::vector<T>> mat(newMatrix, shape);
    return mat;
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
      
      int multi = m1Shape[i] / m2Shape[i];
      range *= multi;
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

    Mat<std::vector<float>> mat(newVector, mat1.shape());
    return mat;
  };


  Mat<std::vector<float>> mat_mul(Mat<std::vector<float>>& mat1, Mat<std::vector<float>>& mat2) {
    std::vector<int> m1Shape = mat1.shape();
    std::vector<int> m2Shape = mat2.shape();

    int range = 1;
    for (int i = 0; i < 4; i++) {
      if (m1Shape[i] < m2Shape[i]) {
        throw std::invalid_argument("One of the axis from Matrix 1 has a smaller value than the same axis from Matrix 2");
      };
      if (m1Shape[i] != m2Shape[i] && m2Shape[i] != 1) {
        throw std::invalid_argument("The value of Matrix 2's axis must be either 1 or the same to the Matrix 1's");
      };

      int multi = m1Shape[i] / m2Shape[i];
      range *= multi;
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

    Mat<std::vector<float>> mat(newVector, mat1.shape());
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

// Mat<std::vector<T>>& mat, std::vector<int>& indices
  
  Mat<std::vector<float>> Conv2d(Mat<std::vector<float>>, int in_channels, int out_channels, int kernel, int padding, int stride) {

  };
};

#endif

//   template <typename T> std::vector<T> Linear();
//   template <typename T> std::vector<T> MLinear();
//   template <typename T> std::vector<T> Conv2D();

//   template <typename T> std::vector<T> BatchNorm2D();
//   template <typename T> std::vector<T> LayerNorm();
//   template <typename T> std::vector<T> Sigmoid();
//   template <typename T> std::vector<T> ReLU();

//   template <typename T> std::vector<T> SGD();
//   template <typename T> std::vector<T> Adam();

//   template <typename T> std::vector<T> save();
//   template <typename T> std::vector<T> load();

//   multi-thread
//   CUDA
