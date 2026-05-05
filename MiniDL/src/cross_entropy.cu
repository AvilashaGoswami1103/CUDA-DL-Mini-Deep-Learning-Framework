#include "cross_entropy.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void cross_entropy_backward_kernel(
    float* pred,
    float* target,
    float* grad,
    int size,
    int batch_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        grad[idx] = (pred[idx] - target[idx]) / batch_size;
    }
}

float CrossEntropyLoss::forward(
    Tensor& pred,
    Tensor& target,
    int batch_size,
    int num_classes) {

    float* h_pred = new float[pred.size];
    float* h_target = new float[target.size];

    pred.toHost(h_pred);
    target.toHost(h_target);

    float loss = 0.0f;

    for (int i = 0; i < pred.size; i++) {
        loss -= h_target[i] * logf(h_pred[i] + 1e-8f);
    }

    loss /= batch_size;

    delete[] h_pred;
    delete[] h_target;

    return loss;
}

Tensor CrossEntropyLoss::backward(
    Tensor& pred,
    Tensor& target,
    int batch_size) {

    Tensor grad(pred.size, false);

    int threads = 256;
    int blocks = (pred.size + threads - 1) / threads;

    cross_entropy_backward_kernel<<<blocks, threads>>>(
        pred.data,
        target.data,
        grad.data,
        pred.size,
        batch_size
    );

    cudaDeviceSynchronize();

    return grad;
}