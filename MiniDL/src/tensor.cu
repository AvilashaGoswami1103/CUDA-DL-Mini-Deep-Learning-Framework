#include "tensor.h"
#include <cuda_runtime.h>

Tensor::Tensor(int size, bool requires_grad) {
    this->size = size;
    this->requires_grad = requires_grad;

    cudaMalloc(&data, size * sizeof(float));

    if (requires_grad) {
        cudaMalloc(&grad, size * sizeof(float));
        cudaMemset(grad, 0, size * sizeof(float));
    }
    else {
        grad = nullptr;
    }
}

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
}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept {
    size = other.size;
    requires_grad = other.requires_grad;
    data = other.data;
    grad = other.grad;

    // Nullify the source so its destructor doesn't free the memory
    other.data = nullptr;
    other.grad = nullptr;
    other.size = 0;
}

// Move assignment operator
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;

    // Free current memory
    cudaFree(data);
    if (grad) cudaFree(grad);

    // Transfer ownership
    size = other.size;
    requires_grad = other.requires_grad;
    data = other.data;
    grad = other.grad;

    // Nullify source
    other.data = nullptr;
    other.grad = nullptr;
    other.size = 0;

    return *this;
}

void Tensor::fromHost(float* h_data) {
    cudaMemcpy(data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);
}

void Tensor::toHost(float* h_data) {
    cudaMemcpy(h_data, data, size * sizeof(float), cudaMemcpyDeviceToHost);
}

void Tensor::zero_grad() {
    if (requires_grad) {
        cudaMemset(grad, 0, size * sizeof(float));
    }
}

Tensor::~Tensor() {
    cudaFree(data);
    if (grad) cudaFree(grad);
}