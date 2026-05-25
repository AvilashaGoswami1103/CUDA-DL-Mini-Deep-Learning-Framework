#pragma once
#include "layer.h"
#include <cuda_runtime.h>

class BatchNorm : public Layer {
public:
    int   num_features;   // number of features (columns)
    float eps;            // numerical stability
    float momentum;       // for running stats update

    // Learnable parameters — trained via backprop
    Tensor* gamma;        // scale
    Tensor* beta;         // shift

    // Running stats — updated during training, used during inference
    float* running_mean;  // on GPU
    float* running_var;   // on GPU

    // Saved during forward, needed for backward
    float* saved_mean;    // batch mean
    float* saved_var;     // batch variance
    float* saved_xhat;    // normalized output before scale/shift

    bool training;

    BatchNorm(int num_features,
        float eps = 1e-5f,
        float momentum = 0.1f);

    ~BatchNorm();

    void set_training(bool mode) { training = mode; }

    Tensor forward(Tensor& x, int batch_size = 0) override;
};
