#pragma once
#include <cuda_runtime.h>

class Tensor {
public:
    float* data;
    float* grad;
    int size;
    bool requires_grad;

    // ✅ ONLY DECLARE
    Tensor(int size, bool requires_grad = false);
    Tensor(const Tensor& other);

    void fromHost(float* h_data);
    void toHost(float* h_data);
    void zero_grad();

    ~Tensor();
};