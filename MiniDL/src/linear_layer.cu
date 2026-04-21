#include "linear_layer.h"
#include "kernels.h"
#include <cuda_runtime.h>

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
    dim3 blocks((out_features + 15) / 16,
        (batch_size + 15) / 16);
    // computes how many blocks are needed


    // Launch matrix multiplication Kernel
    matmul << <blocks, threads >> > (
        x.data, W->data, out.data,
        batch_size, out_features, in_features
        );
    // x.data → input matrix
    // W->data → weight matrix
    // out.data → output matrix


    // Launch bias addition Kernel
    addBias << <blocks, threads >> > (
        out.data, b->data,
        batch_size, out_features
        );
    // Adds bias to every output row: out[row][col] += b[col]


    cudaDeviceSynchronize();
    return out;
}