#include "softmax.h"
#include <cuda_runtime.h>
#include <math.h>
#include <memory>
#include "autograd_context.h"

#ifdef __INTELLISENSE__
#define __CUDACC__
#include <device_launch_parameters.h>
#endif

__global__ void softmax_kernel(float* input, float* output,
    int batch_size, int num_classes) {
    int row = blockIdx.x;
    if (row < batch_size) {
        float sum = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            float val = expf(input[row * num_classes + j]);
            output[row * num_classes + j] = val;
            sum += val;
        }
        for (int j = 0; j < num_classes; j++)
            output[row * num_classes + j] /= sum;
    }
}

Tensor Softmax::forward(Tensor& x, int batch_size) {

    // No-op deleter: x is logits from main, owned by logits variable there
    auto input_sptr = std::shared_ptr<Tensor>(&x, [](Tensor*) {});

    Tensor out(x.size, false);

    softmax_kernel << <batch_size, 1 >> > (x.data, out.data, batch_size, num_classes);
    cudaDeviceSynchronize();

    if (AutogradContext::grad_enabled) {
        out.prev.push_back(input_sptr);   // ← once, inside the if

        out.backward_fn = [input_sptr](Tensor& grad_out) {
            if (input_sptr->grad == nullptr)
                cudaMalloc(&input_sptr->grad, grad_out.size * sizeof(float));
            cudaMemcpy(input_sptr->grad, grad_out.grad,
                grad_out.size * sizeof(float), cudaMemcpyDeviceToDevice);
            };
    }

    return out;
}
