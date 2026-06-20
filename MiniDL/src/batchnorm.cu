#include "batchnorm.h"
#include "autograd_context.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <memory>

#ifdef __INTELLISENSE__
#define __CUDACC__
#include <device_launch_parameters.h>
#endif

// --- Forward kernel (training) ---
// Computes mean, variance, normalizes, scales and shifts
// One thread per feature (column)
__global__ void batchnorm_forward_kernel(
    float* x, float* out, float* xhat,
    float* mean, float* var,
    float* gamma, float* beta,
    float* running_mean, float* running_var,
    float eps, float momentum,
    int batch_size, int num_features) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_features) return;
	// one thread per feature (column)
    
    // Compute mean for feature j
    float mu = 0.0f;
    for (int i = 0; i < batch_size; i++)
        mu += x[i * num_features + j];
    mu /= batch_size;
    mean[j] = mu;

    // Compute variance for feature j
    float v = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        float diff = x[i * num_features + j] - mu;
        v += diff * diff;
    }
    v /= batch_size;
    var[j] = v;

    // Update running stats
    running_mean[j] = (1.0f - momentum) * running_mean[j] + momentum * mu;
    running_var[j] = (1.0f - momentum) * running_var[j] + momentum * v;

    // Normalize, scale, shift
    float inv_std = 1.0f / sqrtf(v + eps);
    for (int i = 0; i < batch_size; i++) {
        float xh = (x[i * num_features + j] - mu) * inv_std;
        xhat[i * num_features + j] = xh;
        out[i * num_features + j] = gamma[j] * xh + beta[j];
    }

}

// --- Forward kernel (inference) ---
// Uses running stats instead of batch stats
__global__ void batchnorm_inference_kernel(
    float* x, float* out,
    float* running_mean, float* running_var,
    float* gamma, float* beta,
    float eps, int batch_size, int num_features) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_features) return;

    // Normalizes with stored statistics, then applies gamma and beta.
    float inv_std = 1.0f / sqrtf(running_var[j] + eps);
    for (int i = 0; i < batch_size; i++) {
        float xh = (x[i * num_features + j] - running_mean[j]) * inv_std;
        out[i * num_features + j] = gamma[j] * xh + beta[j];
    }
}

// --- Backward kernel ---
// One thread per feature — computes d_gamma, d_beta, d_x
__global__ void batchnorm_backward_kernel(
    float* grad_out, float* xhat, float* x,
    float* mean, float* var,
    float* gamma,
    float* d_gamma, float* d_beta, float* d_x,
    float eps, int batch_size, int num_features) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_features) return;

    float inv_std = 1.0f / sqrtf(var[j] + eps);

    // Computes gradients for parameters and input:
    // d_gamma = sum(grad * xhat) over batch
    float dg = 0.0f;
    // d_beta = sum(grad) over batch
    float db = 0.0f;
    // d_xhat = grad * gamma
    // need sum(d_xhat) and sum(d_xhat * xhat) for d_x
    float sum_dxhat = 0.0f;
    float sum_dxhat_xhat = 0.0f;

    for (int i = 0; i < batch_size; i++) {
        int idx = i * num_features + j;
        float g = grad_out[idx];
        float xh = xhat[idx];
        float dxh = g * gamma[j];

        dg += g * xh;
        db += g;
        sum_dxhat += dxh;
        sum_dxhat_xhat += dxh * xh;
    }

    d_gamma[j] = dg;
    d_beta[j] = db;

    // d_x[i] = (1/N) * inv_std * (N*d_xhat[i] - sum_dxhat - xhat[i]*sum_dxhat_xhat)
    float N = (float)batch_size;
    for (int i = 0; i < batch_size; i++) {
        int   idx = i * num_features + j;
        float dxh = grad_out[idx] * gamma[j];
        d_x[idx] = (1.0f / N) * inv_std *
            (N * dxh - sum_dxhat - xhat[idx] * sum_dxhat_xhat);
    }
}

// --- Constructor ---
BatchNorm::BatchNorm(int num_features, float eps, float momentum)
    : num_features(num_features), eps(eps), momentum(momentum), training(true) {

    // Learnable parameters — initialized to gamma=1, beta=0
    gamma = new Tensor(num_features, true);
    beta = new Tensor(num_features, true);

    float* h_gamma = new float[num_features];
    float* h_beta = new float[num_features];
    for (int i = 0; i < num_features; i++) {
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }
    /*Initializes parameters :
    gamma = 1 (scale starts neutral).
    beta = 0 (no shift initially).*/
    gamma->fromHost(h_gamma);
    beta->fromHost(h_beta);
    delete[] h_gamma;
    delete[] h_beta;

    // Running stats — start at mean=0, var=1
    cudaMalloc(&running_mean, num_features * sizeof(float));
    cudaMalloc(&running_var, num_features * sizeof(float));
    cudaMemset(running_mean, 0, num_features * sizeof(float));

    // Initialize running_var to 1
    float* h_ones = new float[num_features];
    for (int i = 0; i < num_features; i++) h_ones[i] = 1.0f;
    cudaMemcpy(running_var, h_ones,
        num_features * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_ones;

    // Saved buffers — allocated on first forward call
    saved_mean = nullptr;
    saved_var = nullptr;
    saved_xhat = nullptr;
}

BatchNorm::~BatchNorm() {
    delete gamma;
    delete beta;
    cudaFree(running_mean);
    cudaFree(running_var);
    if (saved_mean) cudaFree(saved_mean);
    if (saved_var)  cudaFree(saved_var);
    if (saved_xhat) cudaFree(saved_xhat);
}

// Forward function
Tensor BatchNorm::forward(Tensor& x, int batch_size) {

    //Wraps input in a shared pointer for autograd.
    auto input_sptr = std::shared_ptr<Tensor>(&x, [](Tensor*) {});

    Tensor out(x.size, false);
    out.prev.push_back(input_sptr);

    int threads = 256;
    int blocks = (num_features + threads - 1) / threads;

    if (!training) {
        // Inference — use running stats
        batchnorm_inference_kernel << <blocks, threads >> > (
            x.data, out.data,
            running_mean, running_var,
            gamma->data, beta->data,
            eps, batch_size, num_features);

        cudaDeviceSynchronize();
        return out;
    }

    // Training — allocate saved buffers if needed
    int total = batch_size * num_features;
    if (saved_xhat == nullptr) {
        cudaMalloc(&saved_mean, num_features * sizeof(float));
        cudaMalloc(&saved_var, num_features * sizeof(float));
        cudaMalloc(&saved_xhat, total * sizeof(float));
    }
    // Calls batchnorm_forward_kernel to compute batch stats, normalize, and update running stats.
    batchnorm_forward_kernel << <blocks, threads >> > (
        x.data, out.data, saved_xhat,
        saved_mean, saved_var,
        gamma->data, beta->data,
        running_mean, running_var,
        eps, momentum,
        batch_size, num_features);

    cudaDeviceSynchronize();

    if (AutogradContext::grad_enabled) {

        /*If autograd is enabled :
        Attaches a backward function that :
        Zeros gradients for gamma and beta.
        Calls batchnorm_backward_kernel to compute gradients.
        Copies gradient wrt input back into the input tensor.*/

        BatchNorm* self = this;
        out.backward_fn = [input_sptr, self, batch_size](Tensor& grad_out) {

            Tensor d_x(grad_out.size, false);

            // Zero gamma and beta gradients
            cudaMemset(self->gamma->grad, 0,
                self->num_features * sizeof(float));
            cudaMemset(self->beta->grad, 0,
                self->num_features * sizeof(float));

            int threads = 256;
            int blocks = (self->num_features + threads - 1) / threads;

            batchnorm_backward_kernel << <blocks, threads >> > (
                grad_out.grad, self->saved_xhat, input_sptr->data,
                self->saved_mean, self->saved_var,
                self->gamma->data,
                self->gamma->grad, self->beta->grad, d_x.data,
                self->eps, batch_size, self->num_features);

            cudaDeviceSynchronize();

            if (input_sptr->grad == nullptr)
                cudaMalloc(&input_sptr->grad,
                    d_x.size * sizeof(float));

            cudaMemcpy(input_sptr->grad, d_x.data,
                d_x.size * sizeof(float),
                cudaMemcpyDeviceToDevice);
            };
    }

    return out;
}