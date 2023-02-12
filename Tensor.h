#ifndef TENSOR_H
#define TENSOR_H

namespace NNLib {
  template <class T> class Tensor {
    public:
      T* data = nullptr;

    private:
      size_t m_Size = 0;
      size_t s_Size = 0;
      size_t capacity = 0;
      int* dims;

      void ReAlloc(size_t& newCapacity) {
        T* newBlock = new T[newCapacity];
        if (newCapacity < m_Size) {
          m_Size = newCapacity;
        }
        for (size_t i = 0; i < m_Size; i++) {
          newBlock[i] = data[i];
        }
        delete[] data;
        data = newBlock;
        capacity = newCapacity;
        int specify_size[] = {1, (int)m_Size};
        dims = specify_size;
        s_Size = 2;
      };

      void ReAlloc(T& newData, size_t& newCapacity) {
        T* newBlock = new T[newCapacity];
        m_Size = newCapacity;
        for (size_t i = 0; i < m_Size; i++) {
          newBlock[i] = newData;
        };
        delete[] data;
        data = newBlock;
        capacity = newCapacity;
        int specify_size[] = {1, (int)m_Size};
        dims = specify_size;
        s_Size = 2;
      };

      void ReAlloc(T newData[], size_t& newCapacity, int matShape[], size_t& shape_Size) {
        m_Size = newCapacity;
        capacity = newCapacity;
        delete[] data;
        data = newData;
        dims = matShape;
        m_Size = newCapacity;
        s_Size = shape_Size;
      };

    public:
      Tensor() { ReAlloc(1); };
      Tensor(size_t& n) { ReAlloc(n); };
      Tensor(T& item, size_t& n) { ReAlloc(item, n); };
      Tensor(T matrix[], size_t& m_Size, int matShape[], size_t& s_Size) {
        size_t matrix_Size = 1;
        size_t shape_Size = 0;
        for (int i = 0; i < (int)s_Size; i ++) {
          if (matShape[i] < 1) {
            throw std::invalid_argument("All dimensional axis must be above 1, such as {1, 1, 2, 2}");
          };
          matrix_Size *= matShape[i];
          shape_Size ++;
        };
        
        if (m_Size != matrix_Size) { throw std::invalid_argument("Matrix size does not match specified shape"); };
        if (s_Size != shape_Size) { throw std::invalid_argument("Shape size does not match specified number of dimensions"); };

        ReAlloc(matrix, m_Size, matShape, s_Size);
      };

      int matrix_shape() const { return dims; };

      size_t shape_size() const { return s_Size; };

      size_t matrix_size() const { return m_Size; };

      void reshape(int newShape[], size_t new_s_Size) {
        size_t new_m = 1;
        size_t n_new_s = 0;
        for (int i = 0; i < (int)new_s_Size; i ++) {
          if (newShape[i] < 1) {
            throw std::invalid_argument("All dimensional axis must be above 1, such as {1, 1, 2, 2}");
          };
          new_m *= newShape[i];
          n_new_s ++;
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
}

#endif