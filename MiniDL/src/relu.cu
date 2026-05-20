#include "relu.h"
#include <cuda_runtime.h>
#include <memory>
#ifdef __INTELLISENSE__
#define __CUDACC__
#include <device_launch_parameters.h>
#endif

__global__ void relu_forward_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        output[idx] = (input[idx] > 0.0f) ? input[idx] : 0.0f;
}

__global__ void relu_backward_kernel(float* input, float* d_out,
    float* d_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_input[idx] = (input[idx] > 0.0f) ? d_out[idx] : 0.0f;
}

ReLU::~ReLU() {
    // input_sptr now manages lifetime; raw `input` pointer is just for compat
}

Tensor ReLU::forward(Tensor& x, int batch_size) {

    // FIX Bug 5: use shared_ptr for consistent ownership; one copy, not two.
    auto input_sptr = std::shared_ptr<Tensor>(&x, [](Tensor*) {});
    input = input_sptr.get();

    Tensor out(x.size, false);

    int threads = 256;
    int blocks = (x.size + threads - 1) / threads;

    relu_forward_kernel << <blocks, threads >> > (x.data, out.data, x.size);
    cudaDeviceSynchronize();

    // Graph connection — keeps input alive
    out.prev.push_back(input_sptr);

    // FIX Bug 5: capture shared_ptr so input is alive during backward
    out.backward_fn = [input_sptr](Tensor& grad_out) {

        Tensor grad_input(grad_out.size, false);

        int threads = 256;
        int blocks = (grad_out.size + threads - 1) / threads;

        relu_backward_kernel << <blocks, threads >> > (
            input_sptr->data, grad_out.grad, grad_input.data, grad_out.size);

        cudaDeviceSynchronize();

        if (input_sptr->grad == nullptr)
            cudaMalloc(&input_sptr->grad, grad_input.size * sizeof(float));

        cudaMemcpy(input_sptr->grad, grad_input.data,
            grad_input.size * sizeof(float),
            cudaMemcpyDeviceToDevice);

        input_sptr->backward();
        };

    return out;
}



//#include "relu.h"
//#include <cuda_runtime.h>
//
//__global__ void relu_forward_kernel(
//    float* input,
//    float* output,
//    int size) {
//
//    int idx =
//        blockIdx.x * blockDim.x
//        + threadIdx.x;
//
//    if (idx < size) {
//
//        output[idx] =
//            (input[idx] > 0.0f)
//            ? input[idx]
//            : 0.0f;
//    }
//}
//
//__global__ void relu_backward_kernel(
//    float* input,
//    float* d_out,
//    float* d_input,
//    int size) {
//
//    int idx =
//        blockIdx.x * blockDim.x
//        + threadIdx.x;
//
//    if (idx < size) {
//
//        d_input[idx] =
//            (input[idx] > 0.0f)
//            ? d_out[idx]
//            : 0.0f;
//    }
//}
//
//ReLU::~ReLU() {
//
//    if (input)
//        delete input;
//}
//
//Tensor ReLU::forward(
//    Tensor& x,
//    int batch_size) {
//
//    // Cleanup previous input
//    if (input) {
//
//        delete input;
//        input = nullptr;
//    }
//
//    // Store input
//    input = new Tensor(x);
//
//    // Output tensor
//    Tensor out(x.size, false);
//
//    // Forward kernel
//    int threads = 256;
//
//    int blocks =
//        (x.size + threads - 1)
//        / threads;
//
//    relu_forward_kernel << <
//        blocks,
//        threads
//        >> > (
//            x.data,
//            out.data,
//            x.size
//            );
//
//    cudaDeviceSynchronize();
//
//    // Graph parent
//    Tensor* input_ptr = input;
//
//    out.prev.push_back(
//        std::make_shared<Tensor>(x)
//    );
//
//    // Autograd backward rule
//    out.backward_fn =
//        [input_ptr]
//        (Tensor& grad_out) {
//
//        Tensor grad_input(
//            grad_out.size,
//            false
//        );
//
//        int threads = 256;
//
//        int blocks =
//            (grad_out.size + threads - 1)
//            / threads;
//
//        relu_backward_kernel << <
//            blocks,
//            threads
//            >> > (
//                input_ptr->data,
//                grad_out.grad,
//                grad_input.data,
//                grad_out.size
//                );
//
//        cudaDeviceSynchronize();
//
//        // Propagate gradient
//        if (input_ptr->grad == nullptr) {
//
//            cudaMalloc(
//                &input_ptr->grad,
//                grad_input.size * sizeof(float)
//            );
//        }
//
//        cudaMemcpy(
//            input_ptr->grad,
//            grad_input.data,
//            grad_input.size * sizeof(float),
//            cudaMemcpyDeviceToDevice
//        );
//
//        // Recursive autograd
//        input_ptr->backward();
//        };
//
//    return out;
//}