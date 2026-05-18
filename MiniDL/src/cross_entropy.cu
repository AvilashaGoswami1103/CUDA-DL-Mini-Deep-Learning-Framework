#include "cross_entropy.h"
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

__global__ void cross_entropy_backward_kernel(
    float* pred,
    float* target,
    float* grad,
    int size,
    int batch_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        grad[idx] =
            (pred[idx] - target[idx])
            / batch_size;
    }
}

Tensor CrossEntropyLoss::forward(
    Tensor& pred,
    Tensor& target,
    int batch_size,
    int num_classes) {

    float* h_pred = new float[pred.size];
    float* h_target = new float[target.size];

    pred.toHost(h_pred);
    target.toHost(h_target);

    float h_loss = 0.0f;

    for (int i = 0; i < pred.size; i++) {

        h_loss -=
            h_target[i] *
            logf(h_pred[i] + 1e-8f);
    }

    h_loss /= batch_size;

    delete[] h_pred;
    delete[] h_target;

    // 🔥 CREATE LOSS TENSOR
    Tensor loss(1, true);

    // Copy scalar loss to GPU
    cudaMemcpy(
        loss.data,
        &h_loss,
        sizeof(float),
        cudaMemcpyHostToDevice
    );

    // 🔥 GRAPH CONNECTION
    Tensor* pred_ptr = &pred;

    loss.prev.push_back(
        std::make_shared<Tensor>(pred)
    );

    // 🔥 AUTOGRAD BACKWARD FUNCTION
    loss.backward_fn =
        [pred_ptr, target, batch_size]
        (Tensor& grad_out) mutable {

        Tensor grad_pred(
            pred_ptr->size,
            false
        );

        int threads = 256;

        int blocks =
            (pred_ptr->size + threads - 1)
            / threads;

        cross_entropy_backward_kernel << <
            blocks,
            threads
            >> > (
                pred_ptr->data,
                target.data,
                grad_pred.data,
                pred_ptr->size,
                batch_size
                );

        cudaDeviceSynchronize();

        // Store gradient
        pred_ptr->grad = grad_pred.data;

        // Recursive autograd
        pred_ptr->backward();
        };

    return loss;
}