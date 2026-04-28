#include "relu.h"
#include <cuda_runtime.h>

__global__ void relu_forward_kernel(float* input, float* output, int size) {    // CUDA kernel for forward pass of ReLU
    // input: pointer to input array in GPU memory.
    // output: pointer to output array in GPU memory.
    // size : number of elements.

    // global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // if index is valid, apply ReLU
    if (idx < size) {
        output[idx] = (input[idx] > 0.0f) ? input[idx] : 0.0f;
    }
}

__global__ void relu_backward_kernel(float* input, float* d_out, float* d_input, int size) {    // CUDA kernel for backward pass of ReLU
    // Parameters: 
    // input: original input values.
    // d_out: gradient coming from the next layer(downstream).
    // d_input : gradient wrt input(to be passed backward).
    // size : number of elements.


    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        d_input[idx] = (input[idx] > 0.0f) ? d_out[idx] : 0.0f;
    }
    // Gradient rule for ReLU:
    // If input > 0 → gradient flows through unchanged.
    // If input ≤ 0 → gradient is 0 (ReLU blocks it).
}

Tensor ReLU::forward(Tensor& x) {   // implement forward pass in the ReLU class

    input = &x; // stores pointer to the input tensor

    Tensor out(x.size, false);  // creates an output tensor on GPU

    int threads = 256;
    int blocks = (x.size + threads - 1) / threads;

    // launch forward kernel on GPU
    relu_forward_kernel << <blocks, threads >> > (
        x.data,
        out.data,
        x.size
        );

    cudaDeviceSynchronize();

    return out;
}

Tensor ReLU::backward(Tensor& d_out) {  // implements backward pass in the ReLU class

    Tensor d_input(d_out.size, false);  // creates a tensor to store gradients wrt input

    int threads = 256;
    int blocks = (d_out.size + threads - 1) / threads;

    // launches backward kernel to compute gradients wrt input
    relu_backward_kernel << <blocks, threads >> > (
        input->data,
        d_out.data,
        d_input.data,
        d_out.size
        );

    cudaDeviceSynchronize();

    return d_input;//return gradient tensor
}