#include "cross_entropy.h"
#include <cuda_runtime.h>
#include <math.h>
#include <memory>
#include <autograd_context.h>

#ifdef __INTELLISENSE__
#define __CUDACC__
#include <device_launch_parameters.h>
#endif

// Computes combined softmax+CE gradient: (pred - target) / batch_size
// This is mathematically exact when pred is already softmax output.
__global__ void cross_entropy_backward_kernel(
    float* pred, float* target, float* grad,
    int size, int batch_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        grad[idx] = (pred[idx] - target[idx]) / (float)batch_size;
}

Tensor CrossEntropyLoss::forward(
    Tensor& pred,
    Tensor& target,
    int batch_size,
    int num_classes) {

    // Compute scalar loss on CPU
    float* h_pred = new float[pred.size];
    float* h_target = new float[target.size];

    pred.toHost(h_pred);
    target.toHost(h_target);

    float h_loss = 0.0f;
    for (int i = 0; i < pred.size; i++)
        h_loss -= h_target[i] * logf(h_pred[i] + 1e-8f);
    h_loss /= (float)batch_size;

    delete[] h_pred;
    delete[] h_target;

    // Scalar loss tensor (requires_grad=true so it has a grad buffer)
    Tensor loss(1, true);

    cudaMemcpy(loss.data, &h_loss, sizeof(float), cudaMemcpyHostToDevice);

    // Seed gradient: dL/dL = 1. This is the root of the graph.
    float one = 1.0f;
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
        pred_sptr->backward(grad_pred);
        };

    if (AutogradContext::grad_enabled) {
        out.prev.push_back(input_sptr);
        out.backward_fn = [...](Tensor& grad_out) { ... };
    }

    return loss;
}


