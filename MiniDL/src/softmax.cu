#include "softmax.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(float* input,
    float* output,
    int batch_size,
    int num_classes) {

    int row = blockIdx.x;

    if (row < batch_size) {

        float sum = 0.0f;

        // compute exponentials
        for (int j = 0; j < num_classes; j++) {
            float val = expf(input[row * num_classes + j]);
            output[row * num_classes + j] = val;
            sum += val;
        }

        // normalize
        for (int j = 0; j < num_classes; j++) {
            output[row * num_classes + j] /= sum;
        }
    }
}

Tensor Softmax::forward(Tensor& x, int batch_size) {

    Tensor out(x.size, false);
    out.creator = this;

    softmax_kernel << <batch_size, 1 >> > (
        x.data,
        out.data,
        batch_size,
        num_classes
        );

    cudaDeviceSynchronize();

    return out;
}

// dummy backward (handled by cross entropy)
Tensor Softmax::backward(Tensor& grad, int batch_size) {
    return grad;
}