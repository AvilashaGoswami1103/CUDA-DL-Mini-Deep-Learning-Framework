#include "relu.h"
#include <cuda_runtime.h>
#include <memory>
#include "autograd_context.h"

#ifdef __INTELLISENSE__
#define __CUDACC__
#include <device_launch_parameters.h>
#endif

// CUDA Kernels
__global__ void relu_forward_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        output[idx] = (input[idx] > 0.0f) ? input[idx] : 0.0f;
}
//Forward pass kernel : one thread per element.
//If input > 0 → copy it; else → set to 0. Implements y = max(0, x).

// backward pass kernel
__global__ void relu_backward_kernel(float* input, float* d_out,
    float* d_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // one thread per element
    if (idx < size) // gradient flows through only where input > 0: implements derivative of ReLU
        d_input[idx] = (input[idx] > 0.0f) ? d_out[idx] : 0.0f;
}

ReLU::~ReLU() {}    // Empty destructor (ReLU has no learnable parameters to free).

Tensor ReLU::forward(Tensor& x, int batch_size) {   // forward method: defines the forward computation for ReLU

    // No-op deleter: x is owned by Sequential's all_nodes
    auto input_sptr = std::shared_ptr<Tensor>(&x, [](Tensor*) {});
    // Wraps input tensor in a non-owning smart pointer for autograd tracking.
    input = input_sptr.get();
    // Stores raw pointer in input

    Tensor out(x.size, false);   // allocates output tensor of same size as input

    int threads = 256;  // Launches CUDA kernel with enough threads to cover all elements.
    int blocks = (x.size + threads - 1) / threads;
    // Applies ReLU element-wise on GPU.
    relu_forward_kernel << <blocks, threads >> > (x.data, out.data, x.size);
    cudaDeviceSynchronize();

    if (AutogradContext::grad_enabled) {
        out.prev.push_back(input_sptr);   // If gradient tracking is enabled, record input as a dependency.

        out.backward_fn = [input_sptr](Tensor& grad_out) {
            Tensor grad_input(grad_out.size, false);
            int threads = 256;
            int blocks = (grad_out.size + threads - 1) / threads;
            relu_backward_kernel << <blocks, threads >> > (
                input_sptr->data, grad_out.grad, grad_input.data, grad_out.size);
            cudaDeviceSynchronize();
            /*Defines backward function :
            Allocates gradient tensor for input.
            Launches backward kernel to compute gradient wrt input.
            Synchronizes.*/

            if (input_sptr->grad == nullptr)
                cudaMalloc(&input_sptr->grad, grad_input.size * sizeof(float));
            cudaMemcpy(input_sptr->grad, grad_input.data,
                grad_input.size * sizeof(float), cudaMemcpyDeviceToDevice);
            };
            /*Ensures input tensor has gradient storage.
            Copies computed gradient into input’s grad field.*/
    }

    return out;

}

//Forward kernel : applies max(0, x) element - wise.
//Backward kernel : passes gradient only where input > 0.
//Forward method : launches kernels, synchronizes, and sets up autograd backward function.
//Autograd integration : ensures gradients flow correctly during backpropagation.