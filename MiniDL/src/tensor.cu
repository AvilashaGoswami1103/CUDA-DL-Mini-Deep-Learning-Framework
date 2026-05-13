#include "tensor.h"
#include "layer.h"
#include <cuda_runtime.h>

Tensor::Tensor(int size, bool requires_grad) {
    this->size = size;
    this->requires_grad = requires_grad;
    creator = nullptr;
    prev = nullptr;

    cudaMalloc(&data, size * sizeof(float));

    if (requires_grad) {
        cudaMalloc(&grad, size * sizeof(float));
        cudaMemset(grad, 0, size * sizeof(float));
    }
    else {
        grad = nullptr;
    }
}

// Copy constructor
Tensor::Tensor(const Tensor& other) {

    size = other.size;
    requires_grad = other.requires_grad;
    creator = other.creator;
    prev = other.prev;

    cudaMalloc(&data, size * sizeof(float));
    cudaMemcpy(data,
        other.data,
        size * sizeof(float),
        cudaMemcpyDeviceToDevice);

    if (requires_grad) {

        cudaMalloc(&grad, size * sizeof(float));

        cudaMemcpy(grad,
            other.grad,
            size * sizeof(float),
            cudaMemcpyDeviceToDevice);
    }
    else {
        grad = nullptr;
    }
}

//Copy assignment
Tensor& Tensor::operator=(const Tensor& other) {

    if (this != &other) {

        if (data) cudaFree(data);
        if (grad) cudaFree(grad);

        size = other.size;
        requires_grad = other.requires_grad;
        creator = other.creator;
        prev = other.prev;

        cudaMalloc(&data, size * sizeof(float));

        cudaMemcpy(data,
            other.data,
            size * sizeof(float),
            cudaMemcpyDeviceToDevice);

        if (requires_grad) {

            cudaMalloc(&grad, size * sizeof(float));

            cudaMemcpy(grad,
                other.grad,
                size * sizeof(float),
                cudaMemcpyDeviceToDevice);
        }
        else {
            grad = nullptr;
        }
    }

    return *this;
}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept {

    data = other.data;
    grad = other.grad;

    size = other.size;
    requires_grad = other.requires_grad;
    creator = other.creator;
    prev = other.prev;

    other.data = nullptr;
    other.grad = nullptr;
}

// Move assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {

    if (this != &other) {

        if (data) cudaFree(data);
        if (grad) cudaFree(grad);

        data = other.data;
        grad = other.grad;

        size = other.size;
        requires_grad = other.requires_grad;
        creator = other.creator;
        prev = other.prev;

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
    if (requires_grad) {
        cudaMemset(grad, 0, size * sizeof(float));
    }
}

//Tensor Tensor::backward(Tensor& grad, int batch_size) {
//
//    // If tensor has no creator, stop recursion
//    if (creator == nullptr) {
//        return grad;
//    }
//
//    // Call backward of creator layer
//    return creator->backward(grad, batch_size);
//}

Tensor::~Tensor() {

    if (data)
        cudaFree(data);

    if (grad)
        cudaFree(grad);
}

