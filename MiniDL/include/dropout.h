#pragma once
#include "layer.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

class Dropout : public Layer {  // Declares a Dropout class that inherits from Layer
public:
    float p;          // drop probability e.g. 0.5
    float* mask;      // saved mask from forward, lives on GPU
    int    mask_size;
    /*mask is a GPU array storing which activations were kept(1) and which were dropped(0) during forward pass.
    mask_size is the number of elements in the mask(same as input size).*/
    bool   training;  // true = apply dropout, false = pass through

    Dropout(float p = 0.5f);
    // Constructor initializes dropout probability (default 50%).
    ~Dropout();
    // Destructor cleans up GPU memory (mask).

    void set_training(bool mode) { training = mode; }
    // Allows switching between training and inference mode.

    Tensor forward(Tensor& x, int batch_size = 0) override;
    /*Implements the forward pass :
    If training = true :
        Generate a random mask using CURAND.
        Apply mask to input(zero out some activations).
        Scale remaining activations by 1 / (1 - p) so expected output stays consistent.
        If training = false:
    Just return input unchanged.*/
};
