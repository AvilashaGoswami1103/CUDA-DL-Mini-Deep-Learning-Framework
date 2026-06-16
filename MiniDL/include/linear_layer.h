#pragma once
#include <cublas_v2.h>
//C++ header file for NVIDIA’s cuBLAS library, 
//which provides GPU - accelerated implementations of BLAS(Basic Linear Algebra Subprograms).
//enables highly optimized matrix and vector operations—especially GEMM(General Matrix Multiply)
//—on NVIDIA GPUs, forming the computational backbone of deep learning training and inference.

#include "tensor.h"
#include "layer.h"

class Linear : public Layer {

public:
    cublasHandle_t handle;  // GPU context for cuBLAS operations.
    int in_features;
    int out_features;

    Tensor* W;
    Tensor* b;

    // NON-OWNING graph pointer
    Tensor* input;  // A non-owning pointer to the input tensor (used for backpropagation or graph connections).

    Linear(int in_f, int out_f);
    // Constructor initializes weights, bias, and cuBLAS handle. Destructor cleans up resources.
    ~Linear();

    Tensor forward(
        Tensor& x,
        int batch_size
    ) override;
    // implements actual computation: y = xW+b, using cuBLAS for efficient matrix multiplication.
    /*override: The compiler checks that the function signature exactly matches a virtual function in the base class.
    If you mistype the function name or change the parameters, the compiler will throw an error instead of silently creating a new function.*/
};