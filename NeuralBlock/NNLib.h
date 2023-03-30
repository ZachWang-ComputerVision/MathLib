#pragma once

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

namespace NNLib {

    template <class T> class Tensor {
    public:
        T* data;
        int* dims;
    private:
        size_t m_Size = 0;
        size_t s_Size = 0;
        size_t capacity = 0;

        void realloc(size_t new_capacity) {
            T* new_block = new T[new_capacity];
            if (new_capacity < m_Size) {
                m_Size = new_capacity;
            }
            for (size_t i = 0; i < m_Size; i++) {
                new_block[i] = data[i];
            }
            delete[] data;
            data = new_block;
            capacity = new_capacity;
            int specify_size[] = { 1, (int)m_Size };
            dims = specify_size;
            s_Size = 2;
        };

        void realloc(T new_data, size_t new_capacity) {
            T* new_block = new T[new_capacity];
            m_Size = new_capacity;
            for (int i = 0; i < (int)m_Size; i++) {
                new_block[i] = new_data;
            };
            delete[] data;
            data = new_block;
            capacity = new_capacity;
            int specify_size[] = { 1, (int)m_Size };
            dims = specify_size;
            s_Size = 2;
        };

        void realloc(T new_data[], size_t new_capacity, int mat_shape[], size_t shape_Size) {
            m_Size = new_capacity;
            capacity = new_capacity;
            delete[] data;
            data = new_data;
            dims = mat_shape;
            m_Size = new_capacity;
            s_Size = shape_Size;
        };

    public:
        Tensor() { realloc(1); };
        Tensor(size_t& n) { realloc(n); };
        Tensor(T& item, size_t& n) { realloc(item, n); };
        Tensor(T& matrix[], size_t& m_Size, int& mat_shape[], size_t& s_Size) {
            size_t matrix_Size = 1;
            size_t shape_Size = 0;
            for (int i = 0; i < (int)s_Size; i++) {
                if (mat_shape[i] < 1) {
                    throw std::invalid_argument("All dimensional axis must be above 1, such as {1, 1, 2, 2}");
                };
                matrix_Size *= (size_t)mat_shape[i];
                shape_Size++;
            };

            if (m_Size != matrix_Size) { throw std::invalid_argument("Matrix size does not match specified shape"); };
            if (s_Size != shape_Size) { throw std::invalid_argument("Shape size does not match specified number of dimensions"); };

            realloc(matrix, m_Size, mat_shape, s_Size);
        };

        T* get_data() { return data; };
        int* matrix_shape() const { return dims; };

        size_t shape_size() const { return s_Size; };

        size_t matrix_size() const { return m_Size; };

        void reshape(int newShape[], size_t new_s_Size) {
            size_t new_m = 1;
            size_t n_new_s = 0;
            for (int i = 0; i < (int)new_s_Size; i++) {
                if (newShape[i] < 1) {
                    throw std::invalid_argument("All dimensional axis must be above 1, such as {1, 1, 2, 2}");
                };
                new_m *= newShape[i];
                n_new_s++;
            }

            if (new_s_Size == n_new_s) {
                s_Size = new_s_Size;
            }
            else {
                throw std::invalid_argument("The new specified shape size does not match the specified shape's size");
            };

            if (m_Size == new_m) {
                dims = newShape;
            }
            else {
                throw std::invalid_argument("The current shape cannot be converted to the new shape");
            };
        };

        T& operator[](size_t index) {
            if (index < 0 || index > m_Size) {
                throw std::invalid_argument("Index is out of bounds");
            };
            return data[index];
        };

        void print() {};

        ~Tensor() {
            delete[] data;
        };
    };

    // template <typename T> Tensor<T> Zeros(int shape[], size_t shape_size) {
    //   int size = 1;
    //   for (int i = 0; i < (int)shape_size; i++) { size *= shape[i]; };

    //   int const const_size = size;
    //   static T vec[const_size];

    //   for (int i = 0; i < size; i++) { vec[i] = (T)0.0; };

    //   Tensor<T> tensor(vec, (size_t)size, shape, shape_size);
    //   return tensor;
    // };

    // template <typename T> Tensor<T> Zeros_like(Tensor<T>& tensor) {
    //   int* shape = tensor.matrix_shape();
    //   size_t shape_size = tensor.shape_size();

    //   int size = 1;
    //   for (int i = 0; i < (int)shape_size; i++) { size *= shape[i]; };

    //   int const const_size = size;
    //   static T new_vec[const_size];

    //   for (int i = 0; i < size; i++) { new_vec[i] = (T)0.0; };

    //   Tensor<T> new_tensor(new_vec, (size_t)size, shape, shape_size);
    //   return new_tensor;
    // };

    // template <typename T> Tensor<T> Ones(int shape[], size_t shape_size) {
    //   int size = 1;
    //   for (int i = 0; i < shape_size; i++) { size *= shape[i]; };

    //   int const const_size = size;
    //   static T vec[const_size];

    //   for (int i = 0; i < size; i++) { vec[i] = (T)1.0; };

    //   Tensor<T> tensor(vec, (size_t)size, shape, shape_size);
    //   return tensor;
    // };

    // template <typename T> Tensor<T> Ones_like(Tensor<T>& tensor) {
    //   int* shape = tensor.matrix_shape();
    //   size_t shape_size = tensor.shape_size();

    //   int size = 1;
    //   for (int i = 0; i < (int)shape_size; i++) { size *= shape[i]; };

    //   int const const_size = size;
    //   static T new_vec[const_size];

    //   for (int i = 0; i < size; i++) { new_vec[i] = (T)1.0; };

    //   Tensor<T> new_tensor(new_vec, (size_t)size, shape, shape_size);
    //   return new_tensor;
    // };

    // Tensor<int> Random_int(int shape[], size_t shape_size) {
    //   int size = 1;
    //   for (int i = 0; i < (int)shape_size; i++) { size *= shape[i]; };

    //   int const const_size = size;
    //   static int vec[const_size];

    //   for (int i = 0; i < size; i++) { vec[i] = std::rand() % 100; };

    //   Tensor<int> tensor(vec, (size_t)size, shape, shape_size);
    //   return tensor;
    // };

    // template <typename T> Tensor<T> Random_decimal(int shape[], size_t shape_size) {
    //   int size = 1;
    //   for (int i = 0; i < (int)shape_size; i++) { size *= shape[i]; };

    //   int const const_size = size;
    //   static T vec[const_size];

    //   for (int i = 0; i < size; i++) { vec[i] = (T)(rand() % 100) / (T)100.0; };

    //   Tensor<T> tensor(vec, (size_t)size, shape, shape_size);
    //   return tensor;
    // };

    // Tensor<bool> Boolean(bool state, int shape[], size_t shape_size) {
    //   int size = 1;
    //   for (int i = 0; i < shape_size; i++) { size *= shape[i]; };

    //   int const const_size = size;
    //   static bool vec[const_size];

    //   for (int i = 0; i < size; i++) { vec[i] = state; };

    //   Tensor<bool> tensor(vec, (size_t)size, shape, shape_size);
    //   return tensor;
    // };


    // template <typename T> Tensor<T> Eye(int shape[], size_t shape_size) {
    //   int last2nd = shape[(int)shape_size - 2];
    //   int last1st = shape[(int)shape_size - 1];
    //   if (last2nd != last1st) { 
    //     throw std::invalid_argument("The inside matrix is not a square"); 
    //   };
    //   int size = 1;
    //   for (int i = 0; i < (int)shape_size; i++) { size *= shape[i]; };

    //   int const const_size = size;
    //   static T vec[const_size];

    //   for (int i = 0; i < size; i++) { vec[i] = (T)0.0; };

    //   for (int i = 0; i < size / (last2nd * last1st); i++) {
    //     for (int j = 0; j < last1st; j++) {
    //       int idx = j + j * last1st + i * last2nd * last1st;
    //       vec[idx] = (T)1.0;
    //     };
    //   };
    //   Tensor<T> tensor(vec, (size_t)size, shape, shape_size);
    //   return tensor;
    // };


    // template <typename T> void Slice_Recur(T& tensor, T& subtensor, int tensor_shape[], 
    //                                         int target_shape[][2], int tensor_idx, int& subtensor_idx, 
    //                                         int span, int shape_idx, int n_dim) {
    //   if (shape_idx == n_dim - 1) {
    //     for (int i = target_shape[shape_idx][0]; i < target_shape[shape_idx][1]; i++) {
    //       subtensor[subtensor_idx] = tensor[tensor_idx + i];
    //       subtensor_idx ++;
    //     };
    //   }
    //   else {
    //     span /= tensor_shape[shape_idx];
    //     for (int i = target_shape[shape_idx][0]; i < target_shape[shape_idx][1]; i++) {
    //       tensor_idx += i * span;
    //       int new_shape_idx = shape_idx + 1;
    //       Slice_Recur(tensor, subtensor, tensor_shape, target_shape, tensor_idx, subtensor_idx,
    //                   span, new_shape_idx, n_dim);
    //     };
    //   };
    // };

    // template <typename T> Tensor<T> Slice(Tensor<T> tensor, int slice_shape[][2], size_t input_shape_size) {
    //   T* data = tensor.get_data();
    //   int* shape = tensor.matrix_shape();
    //   int span = (int)tensor.matrix_size() / shape[0];
    //   size_t tensor_shape_size = tensor.shape_size();
    //   if (input_shape_size < 1 || input_shape_size > (int)tensor.shape_size()) { 
    //     throw std::invalid_argument("Slice dimension does not match target tensor's dimension "); 
    //   };

    //   int new_size = 1;
    //   int new_shape[tensor_shape_size];
    //   for (size_t i = 0; i < input_shape_size; i++) {
    //     int s = slice_shape[i][0];
    //     int e = slice_shape[i][1];
    //     if (e < s) { throw std::invalid_argument("Begining or end index is set wrong."); };
    //     if (s < 0 || e > shape[i]) { throw std::invalid_argument("Index is out of range."); };
    //     int dim_val = e - s + 1;
    //     new_size *= dim_val;
    //     new_shape[i] = dim_val;
    //   };

    //   T sub_vec[new_size];

    //   Slice_Recur(data, sub_vec, shape, slice_shape, 0, 0, span, 0, (int)tensor_shape_size);

    //   Tensor<T> sub_tensor(sub_vec, (size_t)new_size, new_shape, tensor_shape_size);
    //   return sub_tensor;
    // };


    // template <typename T> Tensor<T> Concat(Tensor<T> tensor1, Tensor<T> tensor2, int dim) {
    //   int* shape1 = tensor1.matrix_shape();
    //   int* shape2 = tensor2.matrix_shape();
    //   if (dim < 0 || dim > (int)tensor1.shape_size() - 1) { throw std::invalid_argument("Dimension is out of range."); };

    //   T* data1 = tensor1.get_data();
    //   T* data2 = tensor2.get_data();

    //   size_t size = 1;
    //   size_t new_shape_size = tensor1.shape_size();
    //   int new_shape[new_shape_size];

    //   for (size_t i = 0; i < tensor1.shape_size(); i++) {
    //     if (i != dim) {
    //       if (shape1[i] != shape2[i]) { throw std::invalid_argument("The dimensions of two matrices are not compitable."); };
    //       new_shape[i] = shape1[i];
    //     } 
    //     else {
    //       new_shape[i] = shape1[i] + shape2[i];
    //     };
    //     size *= shape1[i];
    //   };

    //   // chunk indicates how many item in a chunk. Since I am working with concatenation, it means that
    //   // I combine one chunk from matrix 1 and one chunk from matrix 2 to become one chunk of the new matrix. 
    //   // Then, the process repeats until the loop goes over the entire matrices.
    //   int chunk1 = 1;
    //   int chunk2 = 1;
    //   for (size_t i = dim; i < tensor1.shape_size(); i++) {
    //     chunk1 *= shape1[i];
    //     chunk2 *= shape2[i];
    //   };

    //   size_t new_size = 1;
    //   for (size_t i = 0; i < new_shape_size; i++) { new_size *= (size_t)new_shape[i]; };

    //   T new_vec[new_size];

    //   int new_vec_idx = 0;
    //   // The outer loop goes through n chunks. The inner loop runs within the chunk to fill the new matrix.
    //   for (size_t i = 0; i < size / chunk1; i++) {
    //     for (int j = 0; j < chunk1; j++) {
    //       size_t idx = j + i * chunk1;
    //       new_vec[new_vec_idx] = data1[idx];
    //       new_vec_idx++;
    //     };
    //     for (int k = 0; k < chunk2; k++) {
    //       size_t idx = k + i * chunk2;
    //       new_vec[new_vec_idx] = data2[idx];
    //       new_vec_idx++;
    //     };
    //   };
    //   Tensor<T> new_tensor(new_vec, new_size, new_shape, new_shape_size);
    //   return new_tensor;
    // };


    // template <typename T> Tensor<T> Transpose(Tensor<T> tensor, int idxA, int idxB) { 
    //   int* shape = tensor.matrix_shape();
    //   size_t matrix_size = tensor.matrix_size();
    //   size_t shape_size = tensor.shape_size();
    //   T* data = tensor.get_data();

    //   if (idxA < 0 || idxA >= idxB || idxB > (int)shape_size - 1) {
    //     throw std::invalid_argument("Index is out of range.");
    //   };
    //   if (idxB - idxA != 1) { throw std::invalid_argument("Transpose must be applied to next dimension."); };

    //   // The outer loop calculates how many loops we need to go through before the two transpose indices
    //   // The inner loop calculates how many loops we need to go through after the two transpose indices
    //   // 2 x 2 x 5 x 3 x 2 x 2  if we transpose 5 and 3 , the outer loop is 4 = 2 x 2, and the inner loop is 4 = 2 x 2
    //   int outerLoop = shape[0];
    //   int innerLoop = 1;
    //   for (int i = 1; i < idxA; i ++) { outerLoop *= shape[i]; };
    //   for (int i = idxB + 1; i < (int)shape_size; i++) { innerLoop *= shape[i]; };

    //   // The outerChunk is 5 x 3 x 2 x 2 if we use the above example.
    //   // The innerChunk is 2 x 2, which is same to the innerLoop.
    //   int outerChunk = 1;
    //   int innerChunk = innerLoop;
    //   for (int i = idxA; i < shape_size; i++) {
    //     outerChunk *= shape[i];
    //   };

    //   int target = shape[idxB];
    //   shape[idxB] = shape[idxA];
    //   shape[idxA] = target;

    //   // T new_vec[outerLoop * outerChunk];
    //   T new_vec[matrix_size];
    //   int count = 0;
    //   for (int i = 0; i < outerLoop; i++) {
    //     for (int j = 0; j < shape[idxA]; j++) {
    //       for (int k = 0; k < shape[idxB]; k++) {
    //         int h = k + j * shape[idxB] + i * outerChunk;
    //         for (int w = 0; w < innerLoop; w++) {
    //           int idx = w + h * innerChunk;

    //           new_vec[count] = data[idx];
    //           count++;
    //         };
    //       };
    //     };
    //   };

    //   Tensor<T> new_tensor(new_vec, matrix_size, shape, shape_size);
    //   return new_tensor;
    // };


    // template <typename T> Tensor<T> Permute(Tensor<T> tensor, int& idxA, int& idxB) {
    //   Tensor<T> tensor();
    //   return tensor;
    // };


    // // Implement a function for arithmatic calculation

    // template <typename T> void perform_operation(T (*opera)(T, T), T newVector[], 
    //                                              Tensor<T>& m1, Tensor<T>& m2, int dim_prt, 
    //                                              int& idx, int s1, int e1, int s2, int e2) {
    //   int* shape1 = m1.matrix_shape();
    //   int* shape2 = m2.matrix_shape();

    //   if (dim_prt == (int)m1.shape_size() - 1) {
    //     if (shape1[dim_prt] == shape2[dim_prt]) {
    //       for (int i = 0; i < e1 - s1 + 1; i++) {
    //         newVector[idx] = opera(m1.data[i + s1], m2.data[i + s2]);  // Implement a function for arithmatic calculation
    //         idx++;
    //       };
    //     } else {
    //       for (int i = s1; i < e1; i++) {
    //         for (int j = s2; j < e2; j++) {
    //           newVector[idx] = opera(m1.data[i + s1], m2.data[i + s2]);  // Implement a function for arithmatic calculation
    //           idx++;
    //         };
    //       };
    //     };
    //   } else {
    //     int chunk1 = (e1 - s1 + 1) / shape1[dim_prt];
    //     int chunk2 = (e2 - s2 + 1) / shape2[dim_prt];
    //     int next_dim_prt = dim_prt + 1;
    //     if (shape1[dim_prt] == shape2[dim_prt]) {
    //       for (int i = 0; i < shape1[dim_prt]; i ++) {
    //         int newS1 = s1 + i * chunk1;
    //         int newE1 = e1 + i * chunk1;
    //         int newS2 = s2 + i * chunk2;
    //         int newE2 = e2 + i * chunk2;
    //         perform_operation<T>(opera, newVector, m1, m2, next_dim_prt, idx, newS1, newE1, newS2, newE2);
    //       };
    //     }
    //     else {
    //       for (int i = 0; i < shape1[dim_prt]; i ++) {
    //         int newS1 = s1 + i * chunk1;
    //         int newE1 = e1 + i * chunk1;
    //         for (int j = 0; j < shape2[dim_prt]; j ++) {
    //           int newS2 = s2 + i * chunk2;
    //           int newE2 = e2 + i * chunk2;
    //           perform_operation<T>(opera, newVector, m1, m2, next_dim_prt, idx, newS1, newE1, newS2, newE2);
    //         };
    //       };
    //     };
    //   };
    // };

    // void update_shape_n_size(int m1_shape[], size_t m1_shape_size, int m2_shape[], size_t m2_shape_size, 
    //                          int new_shape[], size_t& new_size) {
    //   if (m1_shape_size != m2_shape_size) { 
    //     throw std::invalid_argument("The number of dimension is different between the two matrices."); 
    //   };
    //   if (m1_shape_size < 1) {
    //     throw std::invalid_argument("The matrix cannot be empty."); 
    //   };

    //   for (size_t i = 0; i < m1_shape_size; i++) {
    //     if (m1_shape[i] != m2_shape[i]) {
    //       if (m1_shape[i] != 1 && m2_shape[i] != 1) {
    //         throw std::invalid_argument("The dimensions are incompatible.");
    //       };
    //     };

    //     int n = std::max(m1_shape[i], m2_shape[i]);
    //     new_shape[i] = n;
    //     new_size *= n;
    //   };
    // };


    // template <typename T> Tensor<T> mat_add(Tensor<T>& mat1, Tensor<T>& mat2) {
    //   int* m1_shape = mat1.matrix_shape();
    //   int* m2_shape = mat2.matrix_shape();
    //   size_t size1 = mat1.shape_size();
    //   size_t size2 = mat2.shape_size();
    //   int new_shape[(int)size1];
    //   size_t new_size = 1;
    //   update_shape_n_size(m1_shape, size1, m2_shape,size2, new_shape, new_size);

    //   T* data1 = mat1.get_data();
    //   T* data2 = mat2.get_data();
    //   T new_vec[(int)new_size];

    //   int dim_prt = 0;
    //   int idx = 0; 
    //   int s1 = 0; 
    //   int s2 = 0;

    //   perform_operation<T>([](T a, T b){ return (a + b); }, new_vec, mat1, mat2, dim_prt, idx, s1, (int)size1, s2, (int)size2);
    //   Tensor<T> new_tensor(new_vec, new_size, new_shape, size1);
    //   return new_tensor;
    // };


    // template <typename T> Tensor<T> mat_sub(Tensor<T>& mat1, Tensor<T>& mat2) {
    //   int* m1_shape = mat1.matrix_shape();
    //   int* m2_shape = mat2.matrix_shape();
    //   size_t size1 = mat1.shape_size();
    //   size_t size2 = mat2.shape_size();
    //   int new_shape[(int)size1];
    //   size_t new_size = 1;
    //   update_shape_n_size(m1_shape, size1, m2_shape,size2, new_shape, new_size);

    //   T* data1 = mat1.get_data();
    //   T* data2 = mat2.get_data();
    //   T new_vec[(int)new_size];

    //   int dim_prt = 0;
    //   int idx = 0; 
    //   int s1 = 0; 
    //   int s2 = 0;

    //   perform_operation<T>([](T a, T b){ return (a - b); }, new_vec, mat1, mat2, dim_prt, idx, s1, (int)size1, s2, (int)size2);
    //   Tensor<T> new_tensor(new_vec, new_size, new_shape, size1);
    //   return new_tensor;
    // };


    // template <typename T> Tensor<T> mat_mul(Tensor<T>& mat1, Tensor<T>& mat2) {
    //   int* m1_shape = mat1.matrix_shape();
    //   int* m2_shape = mat2.matrix_shape();
    //   size_t size1 = mat1.shape_size();
    //   size_t size2 = mat2.shape_size();
    //   int new_shape[(int)size1];
    //   size_t new_size = 1;
    //   update_shape_n_size(m1_shape, size1, m2_shape,size2, new_shape, new_size);

    //   T* data1 = mat1.get_data();
    //   T* data2 = mat2.get_data();
    //   T new_vec[(int)new_size];

    //   int dim_prt = 0;
    //   int idx = 0; 
    //   int s1 = 0; 
    //   int s2 = 0;

    //   perform_operation<T>([](T a, T b){ return (a * b); }, new_vec, mat1, mat2, dim_prt, idx, s1, (int)size1, s2, (int)size2);
    //   Tensor<T> new_tensor(new_vec, new_size, new_shape, size1);
    //   return new_tensor;
    // };


    // template <typename T> Tensor<T> mat_div(Tensor<T>& mat1, Tensor<T>& mat2) {
    //   int* m1_shape = mat1.matrix_shape();
    //   int* m2_shape = mat2.matrix_shape();
    //   size_t size1 = mat1.shape_size();
    //   size_t size2 = mat2.shape_size();
    //   int new_shape[(int)size1];
    //   size_t new_size = 1;
    //   update_shape_n_size(m1_shape, size1, m2_shape,size2, new_shape, new_size);

    //   T* data1 = mat1.get_data();
    //   T* data2 = mat2.get_data();
    //   T new_vec[(int)new_size];

    //   int dim_prt = 0;
    //   int idx = 0; 
    //   int s1 = 0; 
    //   int s2 = 0;

    //   perform_operation<T>([](T a, T b){ return (a / b); }, new_vec, mat1, mat2, dim_prt, idx, s1, (int)size1, s2, (int)size2);
    //   Tensor<T> new_tensor(new_vec, new_size, new_shape, size1);
    //   return new_tensor;
    // };


    // template <typename T> Tensor<T> mat_dot(Tensor<T>& mat1, Tensor<T>& mat2) {
    //   Tensor<T> tensor();
    //   return tensor;
    // };


    // template <typename T> Tensor<T> Conv2d(Tensor<T>& tensor, size_t in_channels, int out_channels, int kernel, int padding, int stride) {

    //   Tensor<T> new_tensor();
    //   return new_tensor;
    // };

    // template <typename T> Tensor<T> BatchNorm2D(Tensor<T>& tensor) {
    //   /*
    //   num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None

    //   num_features  – CC from an expected input of size (N, C, H, W)(N,C,H,W)
    //   eps (float) – a value added to the denominator for numerical stability. Default: 1e-5
    //   momentum (float) – the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
    //   affine (bool) – a boolean value that when set to True, this module has learnable affine parameters. Default: True
    //   track_running_stats (bool) – a boolean value that when set to True, this module tracks the running mean and variance, and when set to False, this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None. When these buffers are None, this module always uses batch statistics. in both training and eval modes. Default: True
    //   */
    //   Tensor<T> new_tensor();
    //   return new_tensor;
    // };


    // // template <typename T> void ReLU(Tensor<T>& tensor) {
    // //   T zero = 0.0;
    // //   for (auto& item : mat.data) { item = std::max(zero, item); };
    // // };


    // // template <typename T> void Sigmoid(Tensor<T>& mat) {
    // //   T one = 1.0;
    // //   for (auto& item : mat.data) { item = one / (one + std::exp(-item)); };
    // // };


    // template <typename T> Tensor<T> normalize_imgs(Tensor<T>& mat, std::vector<int>& shift_values) {
    //   Tensor<T> tensor();
    //   return tensor;
    // };

};



// Adam  https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
// WightInit  https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
// AutoGrad  https://github.com/joelgrus/autograd/blob/part06/tests/test_tensor_matmul.py
//           https://github.com/joelgrus/autograd/blob/part06/autograd/tensor.py

// CUDA  https://forums.developer.nvidia.com/t/how-to-call-cuda-function-from-c-file/61986/3
//       https://forums.developer.nvidia.com/t/how-to-use-class-in-cuda-c/61761
// CUDA  From Scratch


//   template <typename T> std::vector<T> Linear();
//   template <typename T> std::vector<T> MLinear();
//   template <typename T> std::vector<T> Conv2D();


//   template <typename T> std::vector<T> LayerNorm();

//   template <typename T> std::vector<T> Adam();

//   template <typename T> std::vector<T> save();
//   template <typename T> std::vector<T> load();

//   multi-thread
//   CUDA
