#pragma once

#include "layer.h"

class ReLU : public Layer {
public:

    Tensor* input = nullptr;

    Tensor forward(Tensor& x, int batch_size = 0) override;

    Tensor backward(Tensor& d_out, int batch_size = 0) override;

    ~ReLU();
};