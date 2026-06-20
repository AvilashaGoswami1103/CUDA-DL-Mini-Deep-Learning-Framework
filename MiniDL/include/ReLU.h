#pragma once

#include "layer.h"

class ReLU : public Layer {
public:

    Tensor* input = nullptr;    // pointer to input tensor, Stored so the layer can use it later (e.g., for backpropagation).
    Tensor forward(Tensor& x, int batch_size = 0) override;
    // For each element in the input tensor, negative values are set to 0, positive values remain unchanged.
    /*Tensor backward(Tensor& d_out, int batch_size = 0) override;*/
    ~ReLU();
};