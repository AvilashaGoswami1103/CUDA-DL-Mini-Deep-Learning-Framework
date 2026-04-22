#include "kernels.h"
#include<cstdio>

__global__ void matmul(float* A, float* B, float* C,
    int M, int N, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row == 0 && col == 0) {
        printf("Kernel running\n");
    }

    if (row < M && col < N) {

        float sum = 0.0f;

        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

__global__ void addBias(float* C, float* b,
    int M, int N) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        C[row * N + col] += b[col];
    }
}

// FOR dW = XT.dY
__global__ void matmul_backward_dW(float* X, float* dY, float* dW,
    int M, int N, int K) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;    //unique index idx
    int total = K * N;  //K = number of rows (input features) x N = number of columns(output neurons)

    if (idx >= total) return;

    // Converts the 1D thread index into 2D coordinates in the dW matrix
    int row = idx / N;
    int col = idx % N;
    // E.g: if N = 4 and idx = 9, then row = 9/4 = 2, col = 9%4 = 1

    float sum = 0.0f;

    for (int i = 0; i < M; i++) {
        // compute dot product for 1 element of dW
        sum += X[i * K + row] * dY[i * N + col];
    }

    dW[row * N + col] = sum;
    // row * N + col converts 2D coordinates back into a flat 1D index
}

// FOR dX = dY.WT
__global__ void matmul_backward_dX(float* dY, float* W, float* dX,
    int M, int N, int K) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * K;

    if (idx >= total) return;

    int row = idx / K;
    int col = idx % K;

    float sum = 0.0f;

    for (int i = 0; i < N; i++) {
        sum += dY[row * N + i] * W[col * N + i];
    }

    dX[row * K + col] = sum;
}

// FOR db = sum(dY)
__global__ void reduce_sum_bias(float* dY, float* db,
    int M, int N) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Each GPU thread gets a unique index col.
    // Here, each thread is responsible for computing one bias gradient for a specific output neuron(column)
    if (col >= N) return;

    float sum = 0.0f;

    for (int i = 0; i < M; i++) {
        sum += dY[i * N + col];
    }
    // Loop over all examples in the batch.
    // For each example i, take the gradient contribution for neuron col(dY[i, col]).
    // Add them up.

    db[col] = sum;
    // Each thread writes one element of db
}
