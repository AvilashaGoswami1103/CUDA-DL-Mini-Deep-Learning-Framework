#pragma once
#include "tensor.h"
#include <vector>
#include <unordered_map>

class Adam {    // Declares the Adam optimizer, which implements the Adam algorithm 
public:
    // hyperparameters
    float lr;   // learning rate
    float beta1;    // exponential decay for first moment
    float beta2;    // exponential decay for 2nd moment
    float eps;      // small constant for numerical stability
    int   t;      // timestep for bias correction

    std::vector<Tensor*> parameters;

    // Per-parameter moment buffers — keyed by raw pointer address
    std::unordered_map<Tensor*, float*> m;  // first moment
    std::unordered_map<Tensor*, float*> v;  // second moment

//m: Maps each parameter to its first moment estimate(running average of gradients).
//v: Maps each parameter to its second moment estimate(running average of squared gradients).

    Adam(float lr = 0.001f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f);
    // Constructor initializes hyperparameters and sets up optimizer state.

    void add_param(Tensor* param);  // Registers a parameter with the optimizer.
    void zero_grad();   // Clears gradients of all parameters (sets them to zero).
    void step();    // Performs the Adam update for each parameter

	~Adam();    // destructor cleans up allocated memory for moment buffers.
};
