//#pragma once
//#include "tensor.h"
//
//class Layer {   // Every neural network layer MUST implement forward and backward
//public:
//    // Derived classes can override this
//    virtual Tensor forward(Tensor& x, int batch_size = 0) = 0;
//    // =0 means pure virtual function -> this makes Layer an abstract class, cannot create a raw Layer.
//    // only derived layers like Linear, ReLU and Softmax implement it.
//    // this creates an equivalent of torch.nn.Module but at a simplified level.
//    virtual Tensor backward(Tensor& grad, int batch_size = 0) = 0;
//
//    // Tensor& x → input tensor passed by reference
//    // Tensor& grad → gradient tensor passed back during backpropagation
//    virtual ~Layer() {}
//    // declares a virtual destructor
//};

// It enforces that every layer must implement forward and backward

#pragma once

#include "tensor.h"

class Layer {

public:

    virtual Tensor forward(
        Tensor& input,
        int batch_size
    ) = 0;

    /*virtual Tensor forward(...) = 0; → This is a pure virtual function, meaning:
    Every subclass must implement its own forward() method.
    The = 0 syntax marks it as abstract(no default implementation).*/

    // Forces derived classes (like LinearLayer, ConvLayer, etc.) to provide their own implementation of forward().
    //Abstraction: This design allows you to define different kinds of layers(linear, convolutional, dropout, etc.) that all share the same interface.
    virtual ~Layer() {}
};