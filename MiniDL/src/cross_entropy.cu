#include "cross_entropy.h"
#include <cuda_runtime.h>
#include <math.h>
#include <memory>
#include <vector>

__global__ void cross_entropy_backward_kernel(
    float* pred, float* target, float* grad,
    int size, int batch_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        grad[idx] = (pred[idx] - target[idx]) / batch_size;
}

Tensor CrossEntropyLoss::forward(
    Tensor& pred, Tensor& target,
    int batch_size, int num_classes) {

    float* h_pred = new float[pred.size];
    float* h_target = new float[target.size];

    pred.toHost(h_pred);
    target.toHost(h_target);

    float h_loss = 0.0f;
    for (int i = 0; i < pred.size; i++)
        h_loss -= h_target[i] * logf(h_pred[i] + 1e-8f);
    h_loss /= batch_size;

    delete[] h_pred;
    delete[] h_target;

    Tensor loss(1, true);
    cudaMemcpy(loss.data, &h_loss, sizeof(float), cudaMemcpyHostToDevice);

    // FIX: own pred via shared_ptr so backward lambda is safe across epochs
    auto pred_sptr = std::shared_ptr<Tensor>(&pred, [](Tensor*) {});

    loss.prev.push_back(pred_sptr);

    // target captured by value (already done correctly before)
    loss.backward_fn =
        [pred_sptr, target, batch_size](Tensor& grad_out) mutable {

        Tensor grad_pred(pred_sptr->size, false);

        int threads = 256;
        int blocks = (pred_sptr->size + threads - 1) / threads;

        cross_entropy_backward_kernel << <blocks, threads >> > (
            pred_sptr->data, target.data, grad_pred.data,
            pred_sptr->size, batch_size);

        cudaDeviceSynchronize();

        if (pred_sptr->grad == nullptr)
            cudaMalloc(&pred_sptr->grad, grad_pred.size * sizeof(float));

        cudaMemcpy(pred_sptr->grad, grad_pred.data,
            grad_pred.size * sizeof(float),
            cudaMemcpyDeviceToDevice);

        pred_sptr->backward();
        };

    return loss;
}
//#include "cross_entropy.h"
//#include <cuda_runtime.h>
//#include <math.h>
//#include <vector>
//
//__global__ void cross_entropy_backward_kernel(
//    float* pred,
//    float* target,
//    float* grad,
//    int size,
//    int batch_size) {
//
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if (idx < size) {
//        grad[idx] =
//            (pred[idx] - target[idx])
//            / batch_size;
//    }
//}
//
//Tensor CrossEntropyLoss::forward(
//    Tensor& pred,
//    Tensor& target,
//    int batch_size,
//    int num_classes) {
//
//    float* h_pred = new float[pred.size];
//    float* h_target = new float[target.size];
//
//    pred.toHost(h_pred);
//    target.toHost(h_target);
//
//    float h_loss = 0.0f;
//
//    for (int i = 0; i < pred.size; i++) {
//
//        h_loss -=
//            h_target[i] *
//            logf(h_pred[i] + 1e-8f);
//    }
//
//    h_loss /= batch_size;
//
//    delete[] h_pred;
//    delete[] h_target;
//
//    // 🔥 CREATE LOSS TENSOR
//    Tensor loss(1, true);
//
//    // Copy scalar loss to GPU
//    cudaMemcpy(
//        loss.data,
//        &h_loss,
//        sizeof(float),
//        cudaMemcpyHostToDevice
//    );
//
//    // 🔥 GRAPH CONNECTION
//    Tensor* pred_ptr = &pred;
//
//    loss.prev.push_back(
//        std::make_shared<Tensor>(pred)
//    );
//
//    // 🔥 AUTOGRAD BACKWARD FUNCTION
//    loss.backward_fn =
//        [pred_ptr, target, batch_size]
//        (Tensor& grad_out) mutable {
//
//        Tensor grad_pred(
//            pred_ptr->size,
//            false
//        );
//
//        int threads = 256;
//
//        int blocks =
//            (pred_ptr->size + threads - 1)
//            / threads;
//
//        cross_entropy_backward_kernel << <
//            blocks,
//            threads
//            >> > (
//                pred_ptr->data,
//                target.data,
//                grad_pred.data,
//                pred_ptr->size,
//                batch_size
//                );
//
//        cudaDeviceSynchronize();
//
//        // Store gradient in prediction tensor
//        if (pred_ptr->grad == nullptr) {
//
//            cudaMalloc(
//                &pred_ptr->grad,
//                grad_pred.size * sizeof(float)
//            );
//        }
//
//        cudaMemcpy(
//            pred_ptr->grad,
//            grad_pred.data,
//            grad_pred.size * sizeof(float),
//            cudaMemcpyDeviceToDevice
//        );
//
//        // Recursive autograd
//        pred_ptr->backward();
//        };
//
//    return loss;
//}