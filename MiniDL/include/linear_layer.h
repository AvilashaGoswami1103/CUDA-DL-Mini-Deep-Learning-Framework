#pragma once
#include "tensor.h"
# include "layer.h"

class Linear : public Layer {
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
    Tensor forward(Tensor& x, int batch_size) override;
    // Performs forward pass of the linear layer
    // Output = x x W+ b

    // Tensor& x: input tensor, batch_size x in_features
    //batch_size: number of input samples

    Tensor backward(Tensor& d_out, int batch_size);
    // Tensor& d_out: This is the gradient of the loss with respect to the layer’s output 
    // (often called ∂L/∂𝑌).It comes from the next layer in the network during backpropagation.
    // int batch_size: The number of samples in the batch. 
};
