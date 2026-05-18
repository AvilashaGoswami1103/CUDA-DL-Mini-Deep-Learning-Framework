#include "relu.h"
#include <cuda_runtime.h>

__global__ void relu_forward_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > 0.0f) ? input[idx] : 0.0f;
    }
}

__global__ void relu_backward_kernel(float* input, float* d_out, float* d_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_input[idx] = (input[idx] > 0.0f) ? d_out[idx] : 0.0f;
    }
}

ReLU::~ReLU() {
    if (input) delete input;
}

Tensor ReLU::forward(Tensor& x, int batch_size) {
    if (input) {
        delete input;
        input = nullptr;
    }
    input = new Tensor(x);   // owns its own GPU copy

    Tensor out(x.size, false);
    /*out.creator = this;
    out.prev = std::make_shared<Tensor>(x);*/
    out.prev.push_back(std::make_shared<Tensor>(x));

    out.backward_fn =
        [input_ptr]
        (Tensor& grad_out) {

        Tensor grad_input(
            grad_out.size,
            false
        );

        int threads = 256;

        int blocks =
            (grad_out.size + threads - 1)
            / threads;

        relu_backward_kernel << <
            blocks,
            threads
            >> > (
                input_ptr->data,
                grad_out.grad,
                grad_input.data,
                grad_out.size
                );

        cudaDeviceSynchronize();

        input_ptr->grad =
            grad_input.data;

        input_ptr->backward();
        };

    Tensor* input_ptr = input;
    ReLU* self = this;

    out.backward_fn =
        [input_ptr, self, batch_size](Tensor& grad_out) {

        Tensor grad_input =
            self->backward(grad_out, batch_size);

        input_ptr->backward(grad_input);
        };

    int threads = 256;
    int blocks = (x.size + threads - 1) / threads;

    relu_forward_kernel<<<blocks, threads>>>(
        x.data,
        out.data,
        x.size
    );

    cudaDeviceSynchronize();
    return out;
}

//Tensor ReLU::backward(Tensor& d_out, int batch_size) {
//    Tensor d_input(d_out.size, false);
//
//    int threads = 256;
//    int blocks = (d_out.size + threads - 1) / threads;
//
//    relu_backward_kernel<<<blocks, threads>>>(
//        input->data,
//        d_out.data,
//        d_input.data,
//        d_out.size
//    );
//
//    cudaDeviceSynchronize();
//    return d_input;
//}