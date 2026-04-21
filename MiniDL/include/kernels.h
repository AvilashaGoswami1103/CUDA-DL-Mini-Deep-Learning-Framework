#pragma once

//__global__: means function runs on GPU, is launched by CPU(host)
__global__ void matmul(float* A, float* B, float* C,
    int M, int N, int K);
// M -> number of rows in A, N -> number of columns in B, K -> shared dimension

__global__ void addBias(float* C, float* b,
    int M, int N);
// GPU kernel that adds a bias vector to a matrix: 
// M -> rows of C, N -> columns of C
//Declaring GPU kernels
