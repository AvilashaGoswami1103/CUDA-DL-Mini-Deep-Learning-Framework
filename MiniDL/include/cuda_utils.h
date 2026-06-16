#pragma once
#pragma once
#include <cuda_runtime.h>
// Brings in CUDA runtime API definitions (like cudaMalloc, cudaMemcpy, etc.).
#include <cstdio>

#define CUDA_CHECK(call) {                                              \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        printf("CUDA Error: %s\n  at %s line %d\n",                    \
               cudaGetErrorString(err), __FILE__, __LINE__);            \
    }                                                                   \
}

//Wraps a CUDA function call(like cudaMalloc, cudaMemcpy, etc.).
//Captures the return value(cudaError_t err).
//Checks if the call succeeded(err != cudaSuccess).
//If there’s an error :
//Prints a human - readable error string(cudaGetErrorString(err)).
//Prints the file name(__FILE__) and line number(__LINE__) where the error occurred.
