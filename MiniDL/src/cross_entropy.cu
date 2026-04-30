#include "cross_entropy.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void cross_entropy_backward_kernel(
    float* pred,
    float* target,
    float* grad,
    int size) {

    // pred: predicted probabilities (softmax output).
    // target: ground‑truth labels(often one‑hot encoded).
    // grad : output gradient array.
    // size : total number of elements(batch_size × num_classes)

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        grad[idx] = pred[idx] - target[idx];
    }
    // Gradient formula: derivative of cross‑entropy loss w.r.t. predictions is simply pred - target
}

float CrossEntropyLoss::forward(
    Tensor& pred,
    Tensor& target,
    int batch_size,
    int num_classes) {
    // Defines the forward pass: computes the scalar loss value

    float* h_pred = new float[pred.size];
    float* h_target = new float[target.size];
    // Allocates host (CPU) arrays to copy data from GPU tensors

    pred.toHost(h_pred);
    target.toHost(h_target);
    // Copies GPU data into CPU arrays for processing

    float loss = 0.0f;  // loss accumulator initialized

    for (int i = 0; i < pred.size; i++) {
        loss -= h_target[i] * logf(h_pred[i] + 1e-8f);
    }
    // Formula: loss = -Σ target[i] * log(pred[i])
    // Adds 1e-8f to avoid log(0) numerical issues

    loss /= batch_size; // averages the loss across the batch

    delete[] h_pred;    // frees temporary CPU arrays
    delete[] h_target;

    // return scalar loss value
    return loss;
}

Tensor CrossEntropyLoss::backward(
    Tensor& pred,
    Tensor& target) {       // Defines backward pass: computes gradient tensor.

    Tensor grad(pred.size, false);  // Allocates gradient tensor on GPU

    int threads = 256;
    int blocks = (pred.size + threads - 1) / threads;

    cross_entropy_backward_kernel << <blocks, threads >> > (
        pred.data,
        target.data,
        grad.data,
        pred.size
        );
    // launches kernel to compute gradients in parallel

    cudaDeviceSynchronize();

    return grad; // return gradient tensor
}

//  Forward pass:
// Copies predictions and targets to CPU.
// Computes scalar cross‑entropy loss : -Σ target * log(pred) / batch_size.

// Backward pass :
// Runs CUDA kernel to compute gradient : grad = pred - target.
// Returns gradient tensor for backpropagation.