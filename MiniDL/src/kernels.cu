#include "kernels.h"

__global__ void matmul(float* A, float* B, float* C,
    int M, int N, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;    
    // computes the row index this thread is responsible for
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // computes the column index

    // A → matrix A in GPU memory (M × K)
    // B → matrix B in GPU memory(K × N)
    // C → output matrix C in GPU memory(M × N)

    if (row < M && col < N) {
        float sum = 0.0f;   // Initializes accumulator

        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        // Row‑major layout:
        // A[row][i] → A[row * K + i]
        // B[i][col] → B[i * N + col]

        C[row * N + col] = sum;

        //each thread handles one dot product
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