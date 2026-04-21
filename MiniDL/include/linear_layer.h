#pragma once
#include "tensor.h"

class Linear {
public:     // Makes all following members accessible outside the class
    int in_features, out_features;
    // Number of input features (columns of input matrix)
    // Number of output features (columns of output matrix)

    // Each row = one data point
    // Each column = one feature(height, weight, pixel value, hidden unit, etc.)
    
    Tensor* W;  
    // Pointer to a Tensor holding the weight matrix: shape W = in_features x out_features
    Tensor* b;
    // Pointer to a Tensor holding the bias vector: shape: b: out_features

    Tensor* input;
    // Stores a pointer to the input tensor passed to forward

    Linear(int in_f, int out_f);    // Linear Layer

    // Forward Pass
    Tensor forward(Tensor& x, int batch_size);
    // Performs forward pass of the linear layer
    // Output = x x W+ b

    // Tensor& x: input tensor, batch_size x in_features
    //batch_size: number of input samples
};
