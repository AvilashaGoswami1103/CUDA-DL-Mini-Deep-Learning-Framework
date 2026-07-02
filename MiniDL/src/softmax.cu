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
    // Each block handles one row (one sample in the batch).
    int row = blockIdx.x;   // row = index of the sample.
    if (row < batch_size) {

        // Step 1: find max value in this row
        float maxval = input[row * num_classes];
        for (int j = 1; j < num_classes; j++)
            maxval = fmaxf(maxval, input[row * num_classes + j]);

        // Step 2: subtract max then exp — values now in range (-inf, 0]
        // so expf result is in range (0, 1] — never overflows
        float sum = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            float val = expf(input[row * num_classes + j] - maxval);
            output[row * num_classes + j] = val;
            sum += val;
        }

        // Step 3: normalize
        for (int j = 0; j < num_classes; j++)
            output[row * num_classes + j] /= sum;   // Produces probabilities that sum to 1.
    }
}
// forward method
Tensor Softmax::forward(Tensor& x, int batch_size) {    

    // No-op deleter: x is logits from main, owned by logits variable there
    auto input_sptr = std::shared_ptr<Tensor>(&x, [](Tensor*) {});
    // non-owning smart pointer for autograd
    Tensor out(x.size, false);  // allocates output tensor of same size

    softmax_kernel << <batch_size, 1 >> > (x.data, out.data, batch_size, num_classes);  // Launches CUDA kernel: one block per row, one thread per block.
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
    /*If gradient tracking is enabled :
    Records input as dependency.
    Defines backward function :
    Allocates gradient storage if needed.
    Copies gradient from output back to input.*/
    return out;
}
