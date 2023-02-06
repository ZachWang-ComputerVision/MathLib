# NNLib

This project is programmed solely with C++. The objective is to build a package similar to OpenMMLab:
https://github.com/open-mmlab

This project does not depend on external libaraies or packages. I implement the basic matrix operations and autograd functions. User can choose either run on CPU or GPU (CUDA). If the program is ran on CPU, some functions may utilize multi-threading to speed up the calculation. My focus is in Computer Vision, so this is not a complete scientific calculation package. The functions that are implemented are the ones I use regularly, but I have plans to continuously develop it.

I use C++ native data types. If you perfer defining your own data type, you still can use this package. Tp use this package, you just need to include the header files. Here are the functions:

- [x] Zeros, Ones, Bool, Random, Eye
- [x] Slice, Reshape, Concatenate, Transpose, Permutate
- [x] Matrix Addition, Subtraction, Multiplication, Division
- [ ] Matrix Dot Product
- [ ] Linear Layer, Multi-Layer Perceptron
- [ ] Convolution2D
- [x] He Weight Initialization
- [x] BatchNorm2D, LayerNorm
- [x] Sigmoid, ReLU
- [x] SGD, Adam
- [x] Autograd
- [ ] Load, Save Weights

- [ ] Multi-Thread
- [x] CUDA

