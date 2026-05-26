#pragma once
#include "layer.h"
#include <cuda_runtime.h>

class Conv2D : public Layer {
public:
    // Filter dimensions
    int in_channels;
    int out_channels;
    int kernel_h;
    int kernel_w;
    int stride;
    int padding;

    // Learnable parameters
    Tensor* filters;   // shape: (out_channels, in_channels, kernel_h, kernel_w)
    Tensor* bias;      // shape: (out_channels)

    // Saved during forward for backward
    int saved_batch;
    int saved_in_h;
    int saved_in_w;
    int saved_out_h;
    int saved_out_w;

    Conv2D(int in_channels,
        int out_channels,
        int kernel_h,
        int kernel_w,
        int stride = 1,
        int padding = 0);

    ~Conv2D();

    // x is flat: (N * C_in * H * W)
    // batch_size = N
    // Must call set_input_dims before forward or pass via constructor
    void set_input_dims(int h, int w) {
        saved_in_h = h;
        saved_in_w = w;
    }

    Tensor forward(Tensor& x, int batch_size = 0) override;
};
