#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <functional>
#include <vector>


class Tensor {
public:
    float* data;
    float* grad;
    int size;
    bool requires_grad;

    /*Layer* creator;
    std::shared_ptr<Tensor> prev;*/

    // Autograd graph
    std::vector<std::shared_ptr<Tensor>> prev;

    // Backward function
    std::function<void(Tensor&)> backward_fn;

    // ✅ ONLY DECLARE
    Tensor(int size, bool requires_grad = false);
    Tensor(const Tensor& other);    // copy constructor
    Tensor& operator=(const Tensor& other);  // Safe Tensor Copy Assignment
    Tensor(Tensor&& other) noexcept;    // move constructor
    Tensor& operator=(Tensor&& other) noexcept;   // move assignment


    void fromHost(float* h_data);
    void toHost(float* h_data);
    void zero_grad();
    /*Tensor backward(Tensor& grad, int batch_size = 0);*/
    void backward(Tensor& incoming_grad);

    ~Tensor();
};