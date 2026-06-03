#pragma once
#pragma once
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call) {                                              \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        printf("CUDA Error: %s\n  at %s line %d\n",                    \
               cudaGetErrorString(err), __FILE__, __LINE__);            \
    }                                                                   \
}
