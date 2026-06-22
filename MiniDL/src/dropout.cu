#include "dropout.h"
#include "autograd_context.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <memory>

#ifdef __INTELLISENSE__
#define __CUDACC__
#include <device_launch_parameters.h>
#endif

// Generates random mask and applies it in one kernel
__global__ void dropout_forward_kernel(
    float* input, float* output, float* mask,
    float p, float scale, int size, unsigned long long seed) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Each thread gets its own RNG state
        curandState state;
        curand_init(seed, idx, 0, &state);

        float r = curand_uniform(&state);  // random float in (0, 1]
        mask[idx] = (r > p) ? 1.0f : 0.0f;
        output[idx] = input[idx] * mask[idx] * scale;
    }
}
//Forward kernel :
//Each thread processes one element of the input.
//Initializes a CURAND RNG state with a seed.
//Generates a random number r in(0, 1].
//If r > p → keep neuron(mask = 1).
//If r <= p → drop neuron(mask = 0).
//Output = input × mask × scale.
//(Scale = 1 / (1 - p) ensures expected value stays consistent.)

__global__ void dropout_backward_kernel(
    float* grad_out, float* mask,
    float* grad_input, float scale, int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        grad_input[idx] = grad_out[idx] * mask[idx] * scale;
}
//Backward kernel :
//Gradient flows only through neurons that were kept(mask = 1).
//Computes grad_input = grad_out × mask × scale.

Dropout::Dropout(float p) : p(p), mask(nullptr), mask_size(0), training(true) {}
// Initializes dropout probability p, sets mask pointer to null, mask size to 0, and training mode to true.

Dropout::~Dropout() {
    if (mask) cudaFree(mask);
}
// Frees GPU memory for mask if allocated.

Tensor Dropout::forward(Tensor& x, int batch_size) {    // defines forward computation

    auto input_sptr = std::shared_ptr<Tensor>(&x, [](Tensor*) {});
    Tensor out(x.size, false);
    out.prev.push_back(input_sptr);
    /*Wraps input in a non - owning smart pointer for autograd.
    Allocates output tensor of same size.
    Records input as dependency for backprop.*/

    // Inference mode — pass straight through, no mask
    if (!training) {
        cudaMemcpy(out.data, x.data,
            x.size * sizeof(float),
            cudaMemcpyDeviceToDevice);  // If not training, just copy input to output (no dropout applied).

        if (AutogradContext::grad_enabled) {
            out.backward_fn = [input_sptr](Tensor& grad_out) {
                if (input_sptr->grad == nullptr)
                    cudaMalloc(&input_sptr->grad,
                        grad_out.size * sizeof(float));
                cudaMemcpy(input_sptr->grad, grad_out.grad,
                    grad_out.size * sizeof(float),
                    cudaMemcpyDeviceToDevice);
                };
        }
        return out;
        // If autograd is enabled, backward just copies gradients through unchanged.
    }

    // Training mode — generate mask and apply
    float scale = 1.0f / (1.0f - p);
    // Compute scaling factor to preserve expected output magnitude.

    // Allocate or reuse mask buffer
    if (mask_size != x.size) {
        if (mask) cudaFree(mask);
        cudaMalloc(&mask, x.size * sizeof(float));
        mask_size = x.size;
    }

    /*Configure CUDA grid / block dimensions.
    Use system clock as random seed.*/

    int threads = 256;
    int blocks = (x.size + threads - 1) / threads;
    // Use clock as seed — different each forward call
    unsigned long long seed = (unsigned long long)clock();

    /*Launch forward kernel to generate mask and apply dropout.
    Synchronize to ensure kernel finishes.*/
    dropout_forward_kernel << <blocks, threads >> > (
        x.data, out.data, mask,
        p, scale, x.size, seed);

    cudaDeviceSynchronize();

    if (AutogradContext::grad_enabled) {

        // Capture mask pointer and scale by value
        float* saved_mask = mask;

        out.backward_fn = [input_sptr, saved_mask, scale](Tensor& grad_out) {

            Tensor grad_input(grad_out.size, false);

            int threads = 256;
            int blocks = (grad_out.size + threads - 1) / threads;

            dropout_backward_kernel << <blocks, threads >> > (
                grad_out.grad, saved_mask,
                grad_input.data, scale, grad_out.size);

            cudaDeviceSynchronize();

            if (input_sptr->grad == nullptr)
                cudaMalloc(&input_sptr->grad,
                    grad_input.size * sizeof(float));

            cudaMemcpy(input_sptr->grad, grad_input.data,
                grad_input.size * sizeof(float),
                cudaMemcpyDeviceToDevice);
            };
    }

    /*If autograd is enabled :
    Capture mask pointer and scale.
    Define backward function : Launch backward kernel to compute gradient wrt input.
    Allocate gradient storage if needed.
    Copy computed gradient into input’s grad.*/

    return out;
}