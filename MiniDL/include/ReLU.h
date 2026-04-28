#pragma once
#include "tensor.h"

class ReLU {
public:
    Tensor* input;

    Tensor forward(Tensor& x);
    Tensor backward(Tensor& d_out);
};