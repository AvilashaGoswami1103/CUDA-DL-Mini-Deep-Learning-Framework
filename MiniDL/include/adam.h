#pragma once
#include "tensor.h"
#include <vector>
#include <unordered_map>

class Adam {
public:
    float lr;
    float beta1;
    float beta2;
    float eps;
    int   t;      // timestep for bias correction

    std::vector<Tensor*> parameters;

    // Per-parameter moment buffers — keyed by raw pointer address
    std::unordered_map<Tensor*, float*> m;  // first moment
    std::unordered_map<Tensor*, float*> v;  // second moment

    Adam(float lr = 0.001f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f);

    void add_param(Tensor* param);
    void zero_grad();
    void step();

    ~Adam();
};
