#include "relu.h"
#include <cuda_runtime.h>
#include <memory>
#include "autograd_context.h"

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

ReLU::~ReLU() {}

Tensor ReLU::forward(Tensor& x, int batch_size) {

    // No-op deleter: x is owned by Sequential's all_nodes
    auto input_sptr = std::shared_ptr<Tensor>(&x, [](Tensor*) {});
    input = input_sptr.get();

    Tensor out(x.size, false);

    int threads = 256;
    int blocks = (x.size + threads - 1) / threads;
    relu_forward_kernel << <blocks, threads >> > (x.data, out.data, x.size);
    cudaDeviceSynchronize();

    if (AutogradContext::grad_enabled) {
        out.prev.push_back(input_sptr);   // ← once, inside the if

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
                grad_input.size * sizeof(float), cudaMemcpyDeviceToDevice);
            };
    }

    return out;

}
