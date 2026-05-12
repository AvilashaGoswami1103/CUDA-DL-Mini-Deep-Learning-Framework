#pragma once
#include <cuda_runtime.h>

class Layer;
class Tensor {
public:
    float* data;
    float* grad;
    int size;
    bool requires_grad;
    Layer* creator;

    // ✅ ONLY DECLARE
    Tensor(int size, bool requires_grad = false);
    Tensor& operator=(const Tensor& other);  // Safe Tensor Copy Assignment
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;


    void fromHost(float* h_data);
    void toHost(float* h_data);
    void zero_grad();
    Tensor backward(Tensor& grad, int batch_size = 0);

    ~Tensor();
};