#include "adam.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cmath>

#ifdef __INTELLISENSE__
#define __CUDACC__
#include <device_launch_parameters.h>
#endif

__global__ void adam_update_kernel(
    float* param, float* grad,
    float* m, float* v,
    float lr, float beta1, float beta2,
    float eps, float bias_corr1, float bias_corr2,
    int size) {
    /*param → pointer to parameter values(weights) on GPU.
    grad → pointer to gradients of those parameters.*/

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Update moments
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad[idx];
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad[idx] * grad[idx];

        // Bias-corrected moments
        float m_hat = m[idx] / bias_corr1;
        float v_hat = v[idx] / bias_corr2;

        // Parameter update
        param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
    /*Each thread updates one parameter element :
    Update m(first moment).
    Update v(second moment).
    Compute bias - corrected estimates m_hat, v_hat.
    Update parameter using Adam’s formula.*/
}

Adam::Adam(float lr, float beta1, float beta2, float eps)   // initializes hyperparameters and sets timestamp t=0
    : lr(lr), beta1(beta1), beta2(beta2), eps(eps), t(0) {
}   // Initializes hyperparameters and sets timestep t = 0.

void Adam::add_param(Tensor* param) {   // Registers a tensor with the optimizer.
    parameters.push_back(param);

    float* m_buf;
    float* v_buf;
    CUDA_CHECK(cudaMalloc(&m_buf, param->size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&v_buf, param->size * sizeof(float)));
    CUDA_CHECK(cudaMemset(m_buf, 0, param->size * sizeof(float)));
    CUDA_CHECK(cudaMemset(v_buf, 0, param->size * sizeof(float)));
    // Allocates GPU memory for moment buffers (m, v) and initializes them to zero.
    m[param] = m_buf;
    v[param] = v_buf;
    // Stores buffers in maps keyed by parameter pointer.
}

void Adam::zero_grad() {
    for (auto param : parameters)
        param->zero_grad();
}   // Clears gradients of all registered parameters.

void Adam::step() {
    t++;  // increment timestep for bias correction

    float bias_corr1 = 1.0f - powf(beta1, (float)t);    // computes bias correction denominators
    float bias_corr2 = 1.0f - powf(beta2, (float)t);

    // Launches CUDA kernel for each parameter tensor to update values in parallel.
    for (auto param : parameters) {
        int threads = 256;
        int blocks = (param->size + threads - 1) / threads;

        adam_update_kernel << <blocks, threads >> > (
            param->data, param->grad,
            m[param], v[param],
            lr, beta1, beta2, eps,
            bias_corr1, bias_corr2,
            param->size);
    }      // Synchronizes after all updates.
    cudaDeviceSynchronize();
}

// frees GPU memory for all moment buffers
Adam::~Adam() {
    for (auto& pair : m) cudaFree(pair.second);
    for (auto& pair : v) cudaFree(pair.second);
}   // Frees GPU memory for all moment buffers when optimizer is destroyed.