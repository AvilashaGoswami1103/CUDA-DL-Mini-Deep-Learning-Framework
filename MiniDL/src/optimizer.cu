#include "optimizer.h"
#ifdef __INTELLISENSE__
#define __CUDACC__
#include <device_launch_parameters.h>
#endif
// include header with SGD class declared

__global__ void sgd_update(float* param, float* grad, float lr, int size) {
    // param: pointer to the parameter values (weights, biases)
    // grad: pointers to gradient dexcents of those params
    // lr: learning rate.    
    // size: number of elements in the parameter tensor.
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // global thread index

    //if index is within bounds, update the param element using the SGD rule
    if (idx < size) {
        param[idx] -= lr * grad[idx];
    }
    // param[𝑖]←param[𝑖]-lr.grad[𝑖]
}

SGD::SGD(float lr) {    //implement step method of SGD
    this->lr = lr;  //constrcutor for the SGD class
}
// this->lr refers to the class member variable, while lr is the constructor argument

void SGD::step(Tensor* param) {
    int threads = 256;
    int blocks = (param->size + threads - 1) / threads;

    sgd_update << <blocks, threads >> > (   
        //launches CUDA kernel sgd_update with calculated grid and block dimensions
        param->data,
        param->grad,
        lr,
        param->size
        );
    //passes in values and Each thread updates one element of the parameter tensor using the SGD rule.

    cudaDeviceSynchronize();
}

void SGD::add_param(Tensor* param) {
    parameters.push_back(param);
}

void SGD::zero_grad() {
    for (auto param : parameters)
        param->zero_grad();
}

void SGD::step() {
    for (auto param : parameters)
        step(param);
}

// Each CUDA thread updates one parameter element
// The SGD class manages the learning rate and provides a step function to apply updates to any Tensor parameter.