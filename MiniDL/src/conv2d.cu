#include "conv2d.h"
#include "cuda_utils.h"
#include "autograd_context.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <cmath>

#ifdef __INTELLISENSE__
#define __CUDACC__
#include <device_launch_parameters.h>
#endif

// -------------------------------------------------------
// Forward kernel
// Each thread computes one output element: out[n][c_out][oh][ow]
// -------------------------------------------------------
__global__ void conv2d_forward_kernel(
    float* input, float* filters, float* bias, float* output,
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int OH, int OW,
    int stride, int padding) {

    // Thread index maps to one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * OH * OW;
    if (idx >= total) return;

    // Decode flat index into (n, c_out, oh, ow)
    int ow = idx % OW;
    int oh = (idx / OW) % OH;
    int c_out = (idx / OW / OH) % C_out;
    int n = idx / OW / OH / C_out;

    float val = bias[c_out];

    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {

                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;

                // Skip if outside input bounds (handles padding)
                if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                    continue;

                int input_idx = n * (C_in * H * W)
                    + c_in * (H * W)
                    + ih * W
                    + iw;

                int filter_idx = c_out * (C_in * KH * KW)
                    + c_in * (KH * KW)
                    + kh * KW
                    + kw;

                val += input[input_idx] * filters[filter_idx];
            }
        }
    }

    output[idx] = val;
}

// -------------------------------------------------------
// dFilter kernel
// Each thread computes gradient for one filter element
// -------------------------------------------------------
__global__ void conv2d_dfilter_kernel(
    float* input, float* grad_out, float* d_filters,
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int OH, int OW,
    int stride, int padding) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * C_in * KH * KW;
    if (idx >= total) return;

    // Decode flat index into (c_out, c_in, kh, kw)
    int kw = idx % KW;
    int kh = (idx / KW) % KH;
    int c_in = (idx / KW / KH) % C_in;
    int c_out = idx / KW / KH / C_in;

    float grad = 0.0f;

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {

                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;

                if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                    continue;

                int grad_idx = n * (C_out * OH * OW)
                    + c_out * (OH * OW)
                    + oh * OW
                    + ow;

                int input_idx = n * (C_in * H * W)
                    + c_in * (H * W)
                    + ih * W
                    + iw;

                grad += grad_out[grad_idx] * input[input_idx];
            }
        }
    }

    d_filters[idx] = grad;
}

// -------------------------------------------------------
// dInput kernel
// Each thread computes gradient for one input element
// -------------------------------------------------------
__global__ void conv2d_dinput_kernel(
    float* grad_out, float* filters, float* d_input,
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int OH, int OW,
    int stride, int padding) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_in * H * W;
    if (idx >= total) return;

    // Decode flat index into (n, c_in, ih, iw)
    int iw = idx % W;
    int ih = (idx / W) % H;
    int c_in = (idx / W / H) % C_in;
    int n = idx / W / H / C_in;

    float grad = 0.0f;

    for (int c_out = 0; c_out < C_out; c_out++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {

                // Which output position used this input?
                int oh = ih + padding - kh;
                int ow = iw + padding - kw;

                // Must be divisible by stride
                if (oh % stride != 0 || ow % stride != 0)
                    continue;

                oh /= stride;
                ow /= stride;

                if (oh < 0 || oh >= OH || ow < 0 || ow >= OW)
                    continue;

                int grad_idx = n * (C_out * OH * OW)
                    + c_out * (OH * OW)
                    + oh * OW
                    + ow;

                int filter_idx = c_out * (C_in * KH * KW)
                    + c_in * (KH * KW)
                    + kh * KW
                    + kw;

                grad += grad_out[grad_idx] * filters[filter_idx];
            }
        }
    }

    d_input[idx] = grad;
}

// -------------------------------------------------------
// dBias kernel — one thread per output channel
// -------------------------------------------------------
__global__ void conv2d_dbias_kernel(
    float* grad_out, float* d_bias,
    int N, int C_out, int OH, int OW) {

    int c_out = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_out >= C_out) return;

    float grad = 0.0f;
    for (int n = 0; n < N; n++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++)
                grad += grad_out[n * (C_out * OH * OW)
                + c_out * (OH * OW)
                + oh * OW + ow];

    d_bias[c_out] = grad;
}

// -------------------------------------------------------
// Constructor
// -------------------------------------------------------
Conv2D::Conv2D(int in_channels, int out_channels,
    int kernel_h, int kernel_w,
    int stride, int padding)
    : in_channels(in_channels), out_channels(out_channels),
    kernel_h(kernel_h), kernel_w(kernel_w),
    stride(stride), padding(padding),
    saved_batch(0), saved_in_h(0), saved_in_w(0),
    saved_out_h(0), saved_out_w(0) {

    int filter_size = out_channels * in_channels * kernel_h * kernel_w;

    filters = new Tensor(filter_size, true);
    bias = new Tensor(out_channels, true);

    // He initialization — good default for conv layers
    float* h_f = new float[filter_size];
    float  std = sqrtf(2.0f / (in_channels * kernel_h * kernel_w));

    srand((unsigned int)time(0));
    for (int i = 0; i < filter_size; i++)
        h_f[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * std;

    float* h_b = new float[out_channels];
    for (int i = 0; i < out_channels; i++) h_b[i] = 0.0f;

    filters->fromHost(h_f);
    bias->fromHost(h_b);
    delete[] h_f;
    delete[] h_b;
}

Conv2D::~Conv2D() {
    delete filters;
    delete bias;
}

// -------------------------------------------------------
// Forward
// -------------------------------------------------------
Tensor Conv2D::forward(Tensor& x, int batch_size) {

    int N = batch_size;
    int C_in = in_channels;
    int H = saved_in_h;
    int W = saved_in_w;

    // Compute output spatial dims
    int OH = (H - kernel_h + 2 * padding) / stride + 1;
    int OW = (W - kernel_w + 2 * padding) / stride + 1;

    saved_batch = N;
    saved_out_h = OH;
    saved_out_w = OW;

    int out_size = N * out_channels * OH * OW;

    auto input_sptr = std::shared_ptr<Tensor>(&x, [](Tensor*) {});

    Tensor out(out_size, false);
    out.prev.push_back(input_sptr);

    int threads = 256;
    int blocks = (out_size + threads - 1) / threads;

    conv2d_forward_kernel << <blocks, threads >> > (
        x.data, filters->data, bias->data, out.data,
        N, C_in, H, W,
        out_channels, kernel_h, kernel_w,
        OH, OW, stride, padding);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Conv2D forward error: %s\n", cudaGetErrorString(err));

    if (AutogradContext::grad_enabled) {

        Conv2D* self = this;

        out.backward_fn = [input_sptr, self,
            N, C_in, H, W, OH, OW]
            (Tensor& grad_out) {

            // --- dFilters ---
            int filter_size = self->out_channels * C_in
                * self->kernel_h * self->kernel_w;
            int blocks_f = (filter_size + 255) / 256;

            cudaMemset(self->filters->grad, 0,
                filter_size * sizeof(float));

            conv2d_dfilter_kernel << <blocks_f, 256 >> > (
                input_sptr->data, grad_out.grad,
                self->filters->grad,
                N, C_in, H, W,
                self->out_channels,
                self->kernel_h, self->kernel_w,
                OH, OW, self->stride, self->padding);

            // --- dBias ---
            int blocks_b = (self->out_channels + 255) / 256;
            cudaMemset(self->bias->grad, 0,
                self->out_channels * sizeof(float));

            conv2d_dbias_kernel << <blocks_b, 256 >> > (
                grad_out.grad, self->bias->grad,
                N, self->out_channels, OH, OW);

            // --- dInput ---
            Tensor d_input(N * C_in * H * W, false);
            int blocks_i = (N * C_in * H * W + 255) / 256;

            conv2d_dinput_kernel << <blocks_i, 256 >> > (
                grad_out.grad, self->filters->data, d_input.data,
                N, C_in, H, W,
                self->out_channels,
                self->kernel_h, self->kernel_w,
                OH, OW, self->stride, self->padding);

            cudaDeviceSynchronize();

            if (input_sptr->grad == nullptr)
                cudaMalloc(&input_sptr->grad,
                    d_input.size * sizeof(float));

            cudaMemcpy(input_sptr->grad, d_input.data,
                d_input.size * sizeof(float),
                cudaMemcpyDeviceToDevice);
            };
    }

    return out;
}