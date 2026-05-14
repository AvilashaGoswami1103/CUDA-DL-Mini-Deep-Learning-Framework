#include "linear_layer.h"
#include "kernels.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

Linear::Linear(int in_f, int out_f) {
    in_features = in_f;
    out_features = out_f;

    W = new Tensor(in_f * out_f, true);
    b = new Tensor(out_f, true);

    float* h_W = new float[in_f * out_f];
    float* h_b = new float[out_f];

    srand(time(0));

    for (int i = 0; i < in_f * out_f; i++)
        h_W[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    for (int i = 0; i < out_f; i++)
        h_b[i] = 0.0f;

    W->fromHost(h_W);
    b->fromHost(h_b);

    delete[] h_W;
    delete[] h_b;
}

Linear::~Linear() {
    delete W;
    delete b;
    if (input) delete input;
}

Tensor Linear::forward(Tensor& x, int batch_size) {
    if (input != nullptr) {
        delete input;
        input = nullptr;
    }
    input = new Tensor(x);

    Tensor out(batch_size * out_features, true);
    out.creator = this;
    out.prev = &x;

    dim3 threads(16, 16);
    dim3 blocks(
        (out_features + threads.x - 1) / threads.x,
        (batch_size + threads.y - 1) / threads.y
    );

    cudaMemset(out.data, 0, batch_size * out_features * sizeof(float));

    matmul<<<blocks, threads>>>(
        x.data, W->data, out.data,
        batch_size, out_features, in_features
    );

    addBias<<<blocks, threads>>>(
        out.data, b->data,
        batch_size, out_features
    );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    return out;
}

Tensor Linear::backward(Tensor& d_out, int batch_size) {
    Tensor dX(batch_size * in_features, false);

    int threads = 256;

    int total_dW = in_features * out_features;
    int blocks_dW = (total_dW + threads - 1) / threads;

    matmul_backward_dW<<<blocks_dW, threads>>>(
        input->data, d_out.data, W->grad,
        batch_size, out_features, in_features
    );

    int total_dX = batch_size * in_features;
    int blocks_dX = (total_dX + threads - 1) / threads;

    matmul_backward_dX<<<blocks_dX, threads>>>(
        d_out.data, W->data, dX.data,
        batch_size, out_features, in_features
    );

    int blocks_db = (out_features + threads - 1) / threads;
    reduce_sum_bias<<<blocks_db, threads>>>(
        d_out.data, b->grad,
        batch_size, out_features
    );

    cudaDeviceSynchronize();

    return dX;
}