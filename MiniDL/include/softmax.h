#pragma once
#include "tensor.h"

class Softmax {
public:
    int num_classes;

    Softmax(int num_classes) : num_classes(num_classes) {}

    Tensor forward(Tensor& x, int batch_size);

    Tensor backward(Tensor& grad, int batch_size = 0);
};