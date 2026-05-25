#pragma once
#include "layer.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

class Dropout : public Layer {
public:
    float p;          // drop probability e.g. 0.5
    float* mask;      // saved mask from forward, lives on GPU
    int    mask_size;
    bool   training;  // true = apply dropout, false = pass through

    Dropout(float p = 0.5f);
    ~Dropout();

    void set_training(bool mode) { training = mode; }

    Tensor forward(Tensor& x, int batch_size = 0) override;
};
