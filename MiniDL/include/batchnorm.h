#pragma once
#include "layer.h"
#include <cuda_runtime.h>

class BatchNorm : public Layer {    // BatchNorm inherits from Layer
public:
    int   num_features;   // number of features (columns)
    float eps;            // numerical stability
    float momentum;       // for running stats update
    /*Controls how quickly running statistics(mean / variance) are updated.
    High momentum → running stats change slowly.
    Low momentum → running stats adapt quickly.*/

    // Learnable parameters — trained via backprop
    Tensor* gamma;        // scale
    // Scale parameter (learned during training). After normalization, each feature is multiplied by gamma.
    Tensor* beta;         // shift
    // Shift parameter (learned during training). After scaling, each feature is shifted by beta

    // Running stats — updated during training, used during inference
    float* running_mean;  // on GPU
    // Stores the long-term mean of each feature (on GPU). Used during inference.
    float* running_var;   // on GPU
    // Stores the long-term variance of each feature (on GPU). Used during inference. 
    
    // Saved during forward, needed for backward
    float* saved_mean;    // batch mean
    float* saved_var;     // batch variance
    float* saved_xhat;    // normalized output before scale/shift

    bool training;

    BatchNorm(int num_features,
        float eps = 1e-5f,
        float momentum = 0.1f);
    // Initializes the layer with given number of features, epsilon, and momentum.

    ~BatchNorm();

    void set_training(bool mode) { training = mode; }
    // Switches between training and inference.

    Tensor forward(Tensor& x, int batch_size = 0) override;
};
