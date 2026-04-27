// The model takes inputs, passes them through layers (with weights and biases), and produces predictions (often called logits or outputs).
// These predictions are compared to the targets(ground truth) using a loss function like MSE.
// The weights don’t appear directly in the loss function — they influence the loss through predictions.

#include "loss.h"
#include <cuda_runtime.h>

__global__ void mse_forward_kernel(float* pred, float* target, float* out, int size) {
    // Declares a CUDA kernel with parameters: 
    // pred: pointer to predictions array in GPU memory, target: pointer to target array in GPU memory, out: pointer to output array (stores squared differences), size: number of elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // global thread index -> each thread handles one element of the arrays
    if (idx < size) {
        float diff = pred[idx] - target[idx];
        out[idx] = diff * diff;
    }
    //computes squared for that element and stores it in out
}

__global__ void mse_backward_kernel(float* pred, float* target, float* grad, int size) {      // grad: output array for gradients wrt predictions
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        grad[idx] = 2.0f * (pred[idx] - target[idx]) / size;
    }
    // computes gradient of MSE wrt predictions
}

float MSELoss::forward(Tensor& pred, Tensor& target) {      // Implements forwad pass of MSELoss

    Tensor temp(pred.size, false);
    // Creates a temporary tensor on GPU to store squared differences

    int threads = 256;
    int blocks = (pred.size + threads - 1) / threads;

    mse_forward_kernel << <blocks, threads >> > (
        pred.data, target.data, temp.data, pred.size
        ); //launch forward kernel on GPU

    cudaDeviceSynchronize();

    float* h_temp = new float[pred.size];
    temp.toHost(h_temp);
    // copies GPU results back to CPU memory

    // sums all squared differences
    float loss = 0.0f;
    for (int i = 0; i < pred.size; i++)
        loss += h_temp[i];

    loss /= pred.size;
    // Divides by number of elements → mean squared error

    delete[] h_temp;

    return loss;
    // cleans up host memory and returns the loss value
}

Tensor MSELoss::backward(Tensor& pred, Tensor& target) {    // implements backward pass

    Tensor grad(pred.size, false); // creates a tensor to store gradients on GPU

    int threads = 256;
    int blocks = (pred.size + threads - 1) / threads;

    mse_backward_kernel << <blocks, threads >> > (
        pred.data, target.data, grad.data, pred.size
        );  // laucnhes backward kernel to compute gradients wrt predictions

    cudaDeviceSynchronize();

    return grad;
}