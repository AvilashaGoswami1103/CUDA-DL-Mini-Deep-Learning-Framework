#pragma once
#include "tensor.h"


class Softmax {
public:
    Tensor* output;

    Tensor forward(Tensor& x, int batch_size, int num_classes);
    Tensor backward(Tensor& grad, int batch_size = 0) override;
};
