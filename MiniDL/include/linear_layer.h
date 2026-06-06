#pragma once
#include <cublas_v2.h>

#include "tensor.h"
#include "layer.h"

class Linear : public Layer {

public:
    cublasHandle_t handle;
    int in_features;
    int out_features;

    Tensor* W;
    Tensor* b;

    // NON-OWNING graph pointer
    Tensor* input;

    Linear(int in_f, int out_f);

    ~Linear();

    Tensor forward(
        Tensor& x,
        int batch_size
    ) override;
};