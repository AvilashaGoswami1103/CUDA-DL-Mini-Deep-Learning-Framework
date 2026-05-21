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

    // Autograd graph — holds previous nodes alive
    std::vector<std::shared_ptr<Tensor>> prev;

    // Local gradient function set during forward pass
    std::function<void(Tensor&)> backward_fn;

    Tensor(int size, bool requires_grad = false);
    Tensor(const Tensor& other);             // copy — value only, no graph
    Tensor& operator=(const Tensor& other);  // copy assign — value only, no graph
    Tensor(Tensor&& other) noexcept;         // move — transfers graph
    Tensor& operator=(Tensor&& other) noexcept; // move assign — transfers graph

    void fromHost(float* h_data);
    void toHost(float* h_data);
    void zero_grad();

    bool visited = false;

    void free_graph();
    // Called on loss to start backprop. incoming_grad carries dL/d(this).
    void backward();

    ~Tensor();
};
