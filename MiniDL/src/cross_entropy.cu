#include "cross_entropy.h"
#include <cuda_runtime.h>
#include <math.h>
#include <memory>
#include <autograd_context.h>

#ifdef __INTELLISENSE__
#define __CUDACC__
#include <device_launch_parameters.h>
#endif

// Computes element-wise loss: -target[i] * log(pred[i] + eps)
__global__ void cross_entropy_elementwise_kernel(
    float* pred, float* target, float* out,
    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        out[idx] = -target[idx] * logf(pred[idx] + 1e-8f);
}
// computes per-elemment loss
//pred = predicted probabilities(softmax output).
//target = one - hot encoded ground truth.
//out = elementwise loss values.
//1e-8f avoids log(0).

// performs parallel reduction in shared memory -> sums all elements in shared memory
// Reduces all elements to a single sum, then divides by batch_size
__global__ void reduce_sum_kernel(
    float* input, float* output,
    int size, float scale) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(output, sdata[0] * scale);
}
//Divides by batch size(scale = 1 / batch_size).
//Produces a single scalar loss value.

// Computes combined softmax+CE gradient: (pred - target) / batch_size
// This is mathematically exact when pred is already softmax output.
__global__ void cross_entropy_backward_kernel(
    float* pred, float* target, float* grad,
    int size, int batch_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        grad[idx] = (pred[idx] - target[idx]) / (float)batch_size;
}
// computes gradient wrt predictions.
// this is the fued Softmax+CrossEntropy gradient, which is efficient and exact.

Tensor CrossEntropyLoss::forward(
    Tensor& pred,
    Tensor& target,
    int batch_size,
    int num_classes) {  // Defines forward computation for cross-entropy loss.

    // Allocate temp buffer for element-wise loss values
    float* d_elem;
    cudaMalloc(&d_elem, pred.size * sizeof(float));

    // Step 1: compute -target * log(pred) for each element on GPU
    // Launches kernel to compute -target * log(pred) for each element.
    int threads = 256;
    int blocks = (pred.size + threads - 1) / threads;
    cross_entropy_elementwise_kernel << <blocks, threads >> > (
        pred.data, target.data, d_elem, pred.size);

    // Step 2: reduce to scalar and divide by batch_size — all on GPU
    // Reduces elementwise losses to a single scalar.
    Tensor loss(1, true);
    cudaMemset(loss.data, 0, sizeof(float));
    int shared_mem = threads * sizeof(float);
    reduce_sum_kernel << <blocks, threads, shared_mem >> > (
        d_elem, loss.data, pred.size, 1.0f / (float)batch_size);

    cudaDeviceSynchronize();
    cudaFree(d_elem);

    // Seed gradient: dL/dL = 1. This is the root of the graph.
    float one = 1.0f;   // Seeds gradient of loss itself (dL/dL = 1).
    cudaMemcpy(loss.grad, &one, sizeof(float), cudaMemcpyHostToDevice);

    // No-op deleter: pred (softmax output) is owned by `out` in main
    auto pred_sptr = std::shared_ptr<Tensor>(&pred, [](Tensor*) {});
    loss.prev.push_back(pred_sptr);

    // target is captured by value — it's a small CPU-side Tensor, safe to copy
    loss.backward_fn = [pred_sptr, target, batch_size](Tensor& grad_out) mutable {

        // grad_pred holds (softmax_out - target) / batch_size
        Tensor grad_pred(pred_sptr->size, false);

        int threads = 256;
        int blocks = (pred_sptr->size + threads - 1) / threads;

        cross_entropy_backward_kernel << <blocks, threads >> > (
            pred_sptr->data, target.data, grad_pred.data,
            pred_sptr->size, batch_size);

        cudaDeviceSynchronize();

        // Propagate gradient to softmax output node
        if (pred_sptr->grad == nullptr)
            cudaMalloc(&pred_sptr->grad, grad_pred.size * sizeof(float));
        cudaMemcpy(pred_sptr->grad, grad_pred.data,
            grad_pred.size * sizeof(float), cudaMemcpyDeviceToDevice);
        // no backward() call — topo sort handles it
        };
    /*Defines backward function :
    Allocates gradient tensor for predictions.
    Launches backward kernel to compute(pred - target) / batch_size.
    Copies gradient into prediction tensor’s grad.*/

    if (AutogradContext::grad_enabled) {
        
    }

    return loss;
}


