#include "tensor.h"
#include "layer.h"
#include <cuda_runtime.h>
#include <vector>
#include <unordered_set>

Tensor::Tensor(int size, bool requires_grad) {
    this->size = size;
    this->requires_grad = requires_grad;

    cudaMalloc(&data, size * sizeof(float));
    cudaMemset(data, 0, size * sizeof(float));

    if (requires_grad) {
        cudaMalloc(&grad, size * sizeof(float));
        cudaMemset(grad, 0, size * sizeof(float));
    }
    else {
        grad = nullptr;
    }
}

// Copy constructor — copies GPU data only, NO graph transfer.
// A copy is a new value-only leaf node.
Tensor::Tensor(const Tensor& other) {
    size = other.size;
    requires_grad = other.requires_grad;

    cudaMalloc(&data, size * sizeof(float));
    cudaMemcpy(data, other.data, size * sizeof(float), cudaMemcpyDeviceToDevice);

    if (requires_grad && other.grad != nullptr) {
        cudaMalloc(&grad, size * sizeof(float));
        cudaMemcpy(grad, other.grad, size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    else {
        grad = nullptr;
    }
    // backward_fn and prev intentionally NOT copied — copy is a leaf node
}

// Copy assignment — same rule: values only, no graph
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        if (data) cudaFree(data);
        if (grad) cudaFree(grad);

        size = other.size;
        requires_grad = other.requires_grad;

        cudaMalloc(&data, size * sizeof(float));
        cudaMemcpy(data, other.data, size * sizeof(float), cudaMemcpyDeviceToDevice);

        if (requires_grad && other.grad != nullptr) {
            cudaMalloc(&grad, size * sizeof(float));
            cudaMemcpy(grad, other.grad, size * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        else {
            grad = nullptr;
        }
        // backward_fn and prev intentionally NOT copied
    }
    return *this;
}

// Move constructor — transfers EVERYTHING including the graph
Tensor::Tensor(Tensor&& other) noexcept {
    data = other.data;
    grad = other.grad;
    size = other.size;
    requires_grad = other.requires_grad;

    backward_fn = std::move(other.backward_fn);
    prev = std::move(other.prev);

    other.data = nullptr;
    other.grad = nullptr;
}

// Move assignment — transfers EVERYTHING including the graph
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (data) cudaFree(data);
        if (grad) cudaFree(grad);

        data = other.data;
        grad = other.grad;
        size = other.size;
        requires_grad = other.requires_grad;

        backward_fn = std::move(other.backward_fn);
        prev = std::move(other.prev);

        other.data = nullptr;
        other.grad = nullptr;
    }
    return *this;
}

void Tensor::fromHost(float* h_data) {
    cudaMemcpy(data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);
}

void Tensor::toHost(float* h_data) {
    cudaMemcpy(h_data, data, size * sizeof(float), cudaMemcpyDeviceToHost);
}

void Tensor::zero_grad() {
    if (requires_grad && grad != nullptr) {
        cudaMemset(grad, 0, size * sizeof(float));
    }
}

static void build_topo(Tensor* t,
    std::vector<Tensor*>& order,
    std::unordered_set<Tensor*>& visited) {

    if (visited.count(t)) return;
    visited.insert(t);

    for (auto& p : t->prev)
        build_topo(p.get(), order, visited);

    order.push_back(t);
}

void Tensor::backward() {
    // Seed gradient = 1 for the root (loss scalar)
    if (grad == nullptr) {
        cudaMalloc(&grad, size * sizeof(float));
    }
    float one = 1.0f;
    cudaMemcpy(grad, &one, sizeof(float), cudaMemcpyHostToDevice);

    // Build topological order
    std::vector<Tensor*> order;
    std::unordered_set<Tensor*> visited;
    build_topo(this, order, visited);

    // Traverse in reverse — from loss back to inputs
    for (int i = (int)order.size() - 1; i >= 0; i--) {
        Tensor* t = order[i];
        if (t->backward_fn)
            t->backward_fn(*t);
    }
}

Tensor::~Tensor() {
    if (data) cudaFree(data);
    if (grad) cudaFree(grad);
}

void Tensor::free_graph() {
    backward_fn = nullptr;
    prev.clear();
}
