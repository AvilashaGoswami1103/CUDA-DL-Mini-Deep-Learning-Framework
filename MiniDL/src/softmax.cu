#include "softmax.h"
#include <cuda_runtime.h>
#include <math.h>
#include <memory>
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

    // FIX Bug 3: own the input via shared_ptr so the lambda is safe across epochs
    auto input_sptr = std::shared_ptr<Tensor>(&x, [](Tensor*) {});

    Tensor out(x.size, false);

    // Graph connection
    out.prev.push_back(input_sptr);

    // FIX Bug 3: cudaMemcpy the gradient instead of aliasing the pointer
    out.backward_fn = [input_sptr](Tensor& grad_out) {

        if (input_sptr->grad == nullptr)
            cudaMalloc(&input_sptr->grad, input_sptr->size * sizeof(float));

        // Pass gradient through unchanged (softmax/CE combined gradient
        // is already computed by CrossEntropyLoss backward)
        cudaMemcpy(input_sptr->grad, grad_out.grad,
            grad_out.size * sizeof(float),
            cudaMemcpyDeviceToDevice);

        input_sptr->backward();
        };

    softmax_kernel << <batch_size, 1 >> > (x.data, out.data, batch_size, num_classes);
    cudaDeviceSynchronize();

    return out;
}




//#include "softmax.h"
//#include <cuda_runtime.h>
//#include <math.h>
//#include <memory>
//
//__global__ void softmax_kernel(float* input,
//    float* output,
//    int batch_size,
//    int num_classes) {
//
//    int row = blockIdx.x;
//
//    if (row < batch_size) {
//
//        float sum = 0.0f;
//
//        // compute exponentials
//        for (int j = 0; j < num_classes; j++) {
//            float val = expf(input[row * num_classes + j]);
//            output[row * num_classes + j] = val;
//            sum += val;
//        }
//
//        // normalize
//        for (int j = 0; j < num_classes; j++) {
//            output[row * num_classes + j] /= sum;
//        }
//    }
//}
//
//Tensor Softmax::forward(Tensor& x, int batch_size) {
//
//    Tensor out(x.size, false);
//    /*out.creator = this;
//    out.prev = std::make_shared<Tensor>(x);*/
//
//    Tensor* input_ptr = &x;
//
//    out.prev.push_back(std::make_shared<Tensor>(x));
//
//    out.backward_fn =
//        [input_ptr]
//        (Tensor& grad_out) {
//
//        input_ptr->grad =
//            grad_out.grad;
//
//        input_ptr->backward();
//        };
//
//    softmax_kernel << <batch_size, 1 >> > (
//        x.data,
//        out.data,
//        batch_size,
//        num_classes
//        );
//
//    cudaDeviceSynchronize();
//
//    return out;
//}
//
//// dummy backward (handled by cross entropy)
////Tensor Softmax::backward(Tensor& grad, int batch_size) {
////    return grad;
////}