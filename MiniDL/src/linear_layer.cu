#include "linear_layer.h"
#include "kernels.h"
#include <cuda_runtime.h>
#include<cstdio>

//out-of-class defined constructor
Linear::Linear(int in_f, int out_f) {
    in_features = in_f;
    out_features = out_f;

    //Allocate weight and bias parameters
    W = new Tensor(in_f * out_f, true);
    b = new Tensor(out_f, true);

    float* h_W = new float[in_f * out_f];
    // Allocates weight memory on the CPU (temporary)
    float* h_b = new float[out_f];
    // Allocates CPU memory for the bias vector

    for (int i = 0; i < in_f * out_f; i++)
        h_W[i] = 0.01f;
    //initializing all weights to 0.01

    for (int i = 0; i < out_f; i++)
        h_b[i] = 0.0f;
    // Initializes all bias values to zero

    //Copy data to GPU
    W->fromHost(h_W);
    b->fromHost(h_b);

    delete[] h_W;
    delete[] h_b;
}


// Forward Pass
Tensor Linear::forward(Tensor& x, int batch_size) {
    input = &x;   // Stores a pointer to the input tensor

    Tensor out(batch_size * out_features, true);
    //creates the output tensor

    dim3 threads(16, 16);
    //16 threads in X, 16 threads in Y
    dim3 blocks(
        (out_features + threads.x - 1) / threads.x,
        (batch_size + threads.y - 1) / threads.y
    );
    // computes how many blocks are needed

    cudaMemset(out.data, 0, batch_size * out_features * sizeof(float));

    // Launch matrix multiplication Kernel
    matmul <<<blocks, threads>>> (
        x.data, W->data, out.data,
        batch_size, out_features, in_features
        );
    // x.data → input matrix
    // W->data → weight matrix
    // out.data → output matrix


    // Launch bias addition Kernel
    addBias <<<blocks, threads>>> (
        out.data, b->data,
        batch_size, out_features
        );
    // Adds bias to every output row: out[row][col] += b[col]


    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    return out;
}
Tensor Linear::backward(Tensor& d_out, int batch_size) {
    //d_out is the upstream gradient (∂L/∂Y)

    // dX
    Tensor dX(batch_size * in_features, false);
    // Allocate a tensor dX to hold the gradient wrt the input, size: batch_size x in_features

    int threads = 256;   // CUDA threads per block

    // dW
    int total_dW = in_features * out_features;  // calculate how many elements in the weight gradient
    int blocks_dW = (total_dW + threads - 1) / threads; //how many CUDA blocks needed to cover all those elements

    // launch a CUDA kernel to compute dW = Xt.dY
    matmul_backward_dW << <blocks_dW, threads >> > (
        input->data, d_out.data, W->grad,
        batch_size, out_features, in_features
        );
    // input->data = the forward input 𝑋
    // d_out.data = upstream gradient 𝑑𝑌
    // W->grad = where to store ∂L / ∂W

    // dX
    int total_dX = batch_size * in_features;
    int blocks_dX = (total_dX + threads - 1) / threads;

    matmul_backward_dX << <blocks_dX, threads >> > (
        d_out.data, W->data, dX.data,
        batch_size, out_features, in_features
        );
    // d_out.data = upstream gradient
    // W->data = weight matrix(forward weights)
    // dX.data = where to store ∂L / ∂X.

    // db
    int blocks_db = (out_features + threads - 1) / threads;
    //compute how many blocks needed to reduce the bias gradient
    //bias gradient has size out_features

    reduce_sum_bias << <blocks_db, threads >> > (   //Launch a CUDA kernel to compute db = sum(dY) across the batch
        d_out.data, b->grad,    //store result is b->grad
        batch_size, out_features
        );

    cudaDeviceSynchronize();

    return dX;

    // Because dX is needed for propagating gradients to previous layers,
    // while dW and db are stored internally for updating parameters.

    // Each layer receives:

    //d_out = gradient from next layer
    //  and must:

    // 1. Compute gradients of its parameters
    // dW
    // db
    // 2. Compute gradient for previous layer
    // dX
}