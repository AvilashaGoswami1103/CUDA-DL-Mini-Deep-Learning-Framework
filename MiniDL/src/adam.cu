#include "adam.h"
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
}

Adam::Adam(float lr, float beta1, float beta2, float eps)
    : lr(lr), beta1(beta1), beta2(beta2), eps(eps), t(0) {
}

void Adam::add_param(Tensor* param) {
    parameters.push_back(param);

    // Allocate and zero moment buffers on GPU
    float* m_buf;
    float* v_buf;
    cudaMalloc(&m_buf, param->size * sizeof(float));
    cudaMalloc(&v_buf, param->size * sizeof(float));
    cudaMemset(m_buf, 0, param->size * sizeof(float));
    cudaMemset(v_buf, 0, param->size * sizeof(float));

    m[param] = m_buf;
    v[param] = v_buf;
}

void Adam::zero_grad() {
    for (auto param : parameters)
        param->zero_grad();
}

void Adam::step() {
    t++;  // increment timestep

    float bias_corr1 = 1.0f - powf(beta1, (float)t);
    float bias_corr2 = 1.0f - powf(beta2, (float)t);

    for (auto param : parameters) {
        int threads = 256;
        int blocks = (param->size + threads - 1) / threads;

        adam_update_kernel << <blocks, threads >> > (
            param->data, param->grad,
            m[param], v[param],
            lr, beta1, beta2, eps,
            bias_corr1, bias_corr2,
            param->size);
    }
    cudaDeviceSynchronize();
}

Adam::~Adam() {
    for (auto& pair : m) cudaFree(pair.second);
    for (auto& pair : v) cudaFree(pair.second);
}