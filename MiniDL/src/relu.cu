#include "relu.h"
#include <cuda_runtime.h>

__global__ void relu_forward_kernel(
    float* input,
    float* output,
    int size) {

    int idx =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (idx < size) {

        output[idx] =
            (input[idx] > 0.0f)
            ? input[idx]
            : 0.0f;
    }
}

__global__ void relu_backward_kernel(
    float* input,
    float* d_out,
    float* d_input,
    int size) {

    int idx =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (idx < size) {

        d_input[idx] =
            (input[idx] > 0.0f)
            ? d_out[idx]
            : 0.0f;
    }
}

ReLU::~ReLU() {

    if (input)
        delete input;
}

Tensor ReLU::forward(
    Tensor& x,
    int batch_size) {

    // Cleanup previous input
    if (input) {

        delete input;
        input = nullptr;
    }

    // Store input
    input = new Tensor(x);

    // Output tensor
    Tensor out(x.size, false);

    // Forward kernel
    int threads = 256;

    int blocks =
        (x.size + threads - 1)
        / threads;

    relu_forward_kernel << <
        blocks,
        threads
        >> > (
            x.data,
            out.data,
            x.size
            );

    cudaDeviceSynchronize();

    // Graph parent
    Tensor* input_ptr = input;

    out.prev.push_back(
        std::make_shared<Tensor>(x)
    );

    // Autograd backward rule
    out.backward_fn =
        [input_ptr]
        (Tensor& grad_out) {

        Tensor grad_input(
            grad_out.size,
            false
        );

        int threads = 256;

        int blocks =
            (grad_out.size + threads - 1)
            / threads;

        relu_backward_kernel << <
            blocks,
            threads
            >> > (
                input_ptr->data,
                grad_out.grad,
                grad_input.data,
                grad_out.size
                );

        cudaDeviceSynchronize();

        // Propagate gradient
        input_ptr->grad =
            grad_input.data;

        // Recursive autograd
        input_ptr->backward();
        };

    return out;
}