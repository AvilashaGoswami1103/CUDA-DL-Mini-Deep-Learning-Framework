#pragma once
#include "tensor.h"

class Layer {   // Every neural network layer MUST implement forward and backward
public:
    // Derived classes can override this
    virtual Tensor forward(Tensor& x, int batch_size = 0) = 0;
    // =0 means pure virtual function -> this makes Layer an abstract class, cannot create a raw Layer.
    // only derived layers like Linear, ReLU and Softmax implement it.
    // this creates an equivalent of torch.nn.Module but at a simplified level.
    virtual Tensor backward(Tensor& grad, int batch_size = 0) = 0;

    // Tensor& x → input tensor passed by reference
    // Tensor& grad → gradient tensor passed back during backpropagation
    virtual ~Layer() {}
    // declares a virtual destructor
};

// It enforces that every layer must implement forward and backward