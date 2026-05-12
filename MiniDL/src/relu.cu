#include "relu.h"
#include <cuda_runtime.h>

__global__ void relu_forward_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > 0.0f) ? input[idx] : 0.0f;
    }
}

__global__ void relu_backward_kernel(float* input, float* d_out, float* d_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_input[idx] = (input[idx] > 0.0f) ? d_out[idx] : 0.0f;
    }
}

ReLU::~ReLU() {
    if (input) delete input;
}

Tensor ReLU::forward(Tensor& x, int batch_size) {
    if (input != nullptr) {
        delete input;
        input = nullptr;
    }
    input = new Tensor(x);   // owns its own GPU copy

    Tensor out(x.size, false);
    out.creator = this;

    int threads = 256;
    int blocks = (x.size + threads - 1) / threads;

    relu_forward_kernel<<<blocks, threads>>>(
        x.data,
        out.data,
        x.size
    );

    cudaDeviceSynchronize();
    return out;
}

Tensor ReLU::backward(Tensor& d_out, int batch_size) {
    Tensor d_input(d_out.size, false);

    int threads = 256;
    int blocks = (d_out.size + threads - 1) / threads;

    relu_backward_kernel<<<blocks, threads>>>(
        input->data,
        d_out.data,
        d_input.data,
        d_out.size
    );

    cudaDeviceSynchronize();
    return d_input;
}