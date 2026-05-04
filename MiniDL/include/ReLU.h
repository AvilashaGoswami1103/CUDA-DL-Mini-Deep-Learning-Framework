#pragma once

#include "layer.h"

class ReLU : public Layer {
public:

    Tensor* input;

    Tensor forward(Tensor& x, int batch_size = 0) override;

    Tensor backward(Tensor& d_out, int batch_size = 0) override;
};