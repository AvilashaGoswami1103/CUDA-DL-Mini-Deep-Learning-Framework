#pragma once

__global__ void matmul(float* A, float* B, float* C,
    int M, int N, int K);

__global__ void addBias(float* C, float* b,
    int M, int N);

__global__ void matmul_backward_dW(float* X, float* dY, float* dW,
    int M, int N, int K);
// Compute the gradient of the inputs 𝑑𝑋


__global__ void matmul_backward_dX(float* dY, float* W, float* dX,
    int M, int N, int K);
// Compute the gradient of the inputs dX

__global__ void reduce_sum_bias(float* dY, float* db,
    int M, int N);
// Compute the gradient of the bias 𝑑𝑏


//X: input matrix of shape(M × K).
//dY : gradient matrix of shape(M × N).
//dW : output gradient of weights, shape(K × N).
//M : batch size(# of examples).
//N : output dimension(# of neurons).
//K : input dimension(# of features).
