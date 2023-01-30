// #include <iostream>
// #include <vector>

// #include "MathLib.h"

// namespace MathLib {
//   // template <class T> Mat<T>::Mat(T& matrix, T& matShape) {
//   //   mat = matrix;
//   //   shape = matShape;
//   // };

//   // int Mat<int>::size() { return sizeof(shape); };
//   // void Mat<void>::print() {
//   //   // for (int i = 0; i < 3; i++) {
//   //   //   std::cout << "i: " << i << std::endl;
//   //   //   std::cout << *(shape + i) << std::endl;
//   //   // }
//   //   std::cout << "its in" << std::endl;
//   // };

//   template <typename T> std::vector<T> transpose(std::vector<T> matrixint, int shape) { 

//   };

//   template <typename T> std::vector<T> reshape(std::vector<T> matrix, int curShape, int newShape) {
//     int curSize = 1;
//     int newSize = 1;
//     for (int i = 0; i < sizeof(curShape) / 4; i++) { curSize *= curShape[i]; };
//     for (int i = 0; i < sizeof(newShape) / 4; i++) { newSize *= newShape[i]; };
//     if (curSize != newSize) {
//       throw std::invalid_argument("The new shape does not match existing matrix shape.")
//     }
    
//   };
// };

    // 

    // int reshape(int matReshape[]) {
    //   int curShape = 1;
    //   int newShape = 1;

    //   for (int j = 0; j < sizeof(size()) / 4; j++) {
    //     curShape *= curShape[j]
    //   }
    //   for (int i = 0; i < sizeof(matReshape) / 4; i++) {
    //     newShape *= matReshape[i]
    //   }

    //   if (curShape == newShape) {

    //   }
    //   else {

    //   }
      
    // };

//     ~Mat() {};
    
// };

// class MathLib::Zeros {
//   std::vector<float> mat;
//   std::vector<std::vector<float>> mat;
//   std::vector<std::vector<std::vector<float>>> mat;
//   std::vector<std::vector<std::vector<std::vector<float>>>> mat;

//   public:
//     Zeros(int i) {
      
//     };

//     Zeros(int i, int j) {

//     };

    // std::vector<float> mat;
    // int dims = shape.size();

    // if (dims == 0) { return mat; };
    
    // mat.reserve(shape.back());
    // for (int i = 0; i < mat.size(); i++) {
    //     mat[i] = 0.;
    // }

    // if (dims == 1) { return mat; };

    // for (int j = dims - 2; j < 0; j--) {
    //     int dim = shape[j];
    //     std::vector<std::vector<float>> new_mat;
    //     new_mat.reserve(dim);
    //     for (int k = 0; k < dim; k++) { new_mat[k] = mat; };
    //     // mat = new_mat;
    // };

    // return mat;
// };