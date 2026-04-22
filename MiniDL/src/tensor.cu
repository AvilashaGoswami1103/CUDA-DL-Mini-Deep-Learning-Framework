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

    if (requires_grad) {
        cudaMalloc(&grad, size * sizeof(float));
        cudaMemcpy(grad, other.grad, size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    else {
        grad = nullptr;
    }
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