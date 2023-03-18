# Neural Block

This project is programmed solely with C++. The objective is to build a package similar to OpenMMLab:
https://github.com/open-mmlab

This project does not depend on external libaraies or packages. I implement the basic matrix operations and autograd functions. User can choose either run on CPU or GPU (CUDA). If the program is ran on CPU, some functions may utilize multi-threading to speed up the calculation. My focus is in Computer Vision, so this is not a complete scientific calculation package. The functions that are implemented are the ones I use regularly, but I have plans to continuously develop it.

I use C++ native data types. If you perfer defining your own data type, you still can use this package. Tp use this package, you just need to include the header files. Here are the functions:

CPU (single-thread):
- [x] Zeros, Ones, Bool, Random, Eye
- [x] zeros_like, ones_like, bools_like
- [x] Slice, Reshape, Concatenate, Transpose, Permutate
- [x] Matrix Addition, Subtraction, Multiplication, Division
- [ ] Matrix Dot Product
- [ ] Image normalization
- [ ] Linear Layer, Multi-Layer Perceptron
- [ ] Convolution2D
- [ ] He Weight Initialization
- [ ] BatchNorm2D, LayerNorm
- [ ] Sigmoid, ReLU
- [ ] Adam Optimizer
- [ ] Autograd
- [ ] Load, Save Weights

<br>

- [ ] Multi-Thread
- [ ] CUDA



