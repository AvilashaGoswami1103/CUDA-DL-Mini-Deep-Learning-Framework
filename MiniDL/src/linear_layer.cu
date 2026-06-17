#include "linear_layer.h"
#include "kernels.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include "autograd_context.h"

#ifdef __INTELLISENSE__
#define __CUDACC__
#include <device_launch_parameters.h>
#define CUDA_KERNEL(grid, block)
#define <<<grid, block>>>
#endif

Linear::Linear(int in_f, int out_f) {
    cublasCreate(&handle);
    in_features = in_f;
    out_features = out_f;

    W = new Tensor(in_f * out_f, true);
    b = new Tensor(out_f, true);
    // Allocates weight and bias tensors on GPU.

    float* h_W = new float[in_f * out_f];
    float* h_b = new float[out_f];
	// temporary host arrays for initializing weights and biases.

    // Those host arrays(h_W and h_b) are simply temporary arrays allocated in CPU(host) memory that you use to initialize your layer’s parameters before copying them over to the GPU.

    srand((unsigned int)time(0));

    for (int i = 0; i < in_f * out_f; i++)
        h_W[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    // Randomly initializes weights in a small range (centered around 0).

    for (int i = 0; i < out_f; i++)
        h_b[i] = 0.0f;  // initializes biases to zero

    W->fromHost(h_W);
    b->fromHost(h_b);

    delete[] h_W;
    delete[] h_b;

    input = nullptr;
}

Linear::~Linear() {
    cublasDestroy(handle);
    delete W;
    delete b;
}
// Cleans up cuBLAS handle and frees weight / bias tensors.

Tensor Linear::forward(Tensor& x, int batch_size) { // Defines the forward computation for the layer.

    // No-op deleter: x is owned by Sequential's all_nodes vector, not by us
    auto input_sptr = std::shared_ptr<Tensor>(&x, [](Tensor*) {});
    input = input_sptr.get();
    // Wraps input in a shared pointer (non-owning) for autograd tracking.

    Linear* self = this;

    Tensor out(batch_size * out_features, true);
    /*out.prev.push_back(input_sptr);*/

    dim3 threads(16, 16);
    dim3 blocks(
        (out_features + threads.x - 1) / threads.x,
        (batch_size + threads.y - 1) / threads.y
    );
    // Defines CUDA grid/block dimensions for bias addition kernel.

    cudaMemset(out.data, 0, batch_size * out_features * sizeof(float));
	// clears output buffer on GPU to avoid garbage values.

    //cuBLAS Matmul
    // cuBLAS uses column-major. To compute C = A*B (row-major)
// we compute C^T = B^T * A^T using cuBLAS column-major convention.
// alpha=1, beta=0 means C = 1*A*B + 0*C
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        out_features,  // rows of output
        batch_size,    // cols of output
        in_features,   // inner dimension
        &alpha,
        W->data,       // weight matrix
        out_features,  // leading dimension of W
        x.data,        // input matrix
        in_features,   // leading dimension of x
        &beta,
        out.data,      // output
        out_features); // leading dimension of output

    /*Performs matrix multiplication using cuBLAS:
    out = 𝑥⋅ 𝑊*/

    addBias << <blocks, threads >> > (
        out.data, b->data, batch_size, out_features);

    cudaDeviceSynchronize();
    // Launches custom CUDA kernel to add bias to each row, then synchronizes.

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA Error in Linear::forward: %s\n", cudaGetErrorString(err));

    if (AutogradContext::grad_enabled) {

        out.prev.push_back(input_sptr);
        out.backward_fn = [input_sptr, self, batch_size](Tensor& grad_out)  // If gradients are enabled, attaches backward function to output tensor.
            {
            Tensor grad_input(batch_size * self->in_features, false);   // prepares gradient tensor for input.
            int threads = 256;

            cudaMemset(self->W->grad, 0, self->W->size * sizeof(float));
            cudaMemset(self->b->grad, 0, self->b->size * sizeof(float));
            // Clears gradients for weights and bias.

            // Launches CUDA kernels to compute gradients for weights, input, and bias.
            int blocks_dW = (self->in_features * self->out_features + threads - 1) / threads;
            matmul_backward_dW << <blocks_dW, threads >> > (
                input_sptr->data, grad_out.grad, self->W->grad,
                batch_size, self->out_features, self->in_features);

            int blocks_dX = (batch_size * self->in_features + threads - 1) / threads;
            matmul_backward_dX << <blocks_dX, threads >> > (
                grad_out.grad, self->W->data, grad_input.data,
                batch_size, self->out_features, self->in_features);

            int blocks_db = (self->out_features + threads - 1) / threads;
            reduce_sum_bias << <blocks_db, threads >> > (
                grad_out.grad, self->b->grad,
                batch_size, self->out_features);

            cudaDeviceSynchronize();

            if (input_sptr->grad == nullptr)
                cudaMalloc(&input_sptr->grad, grad_input.size * sizeof(float));
            // Synchronizes, allocates gradient storage if needed

            cudaMemcpy(input_sptr->grad, grad_input.data,
                grad_input.size * sizeof(float), cudaMemcpyDeviceToDevice);
            // copies computed input gradient back.
            // no backward() call here — topo sort handles traversal
            };
    }

    return out;
}