#include "linear_layer.h"
#include "kernels.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>

#ifdef __INTELLISENSE__
#define __CUDACC__
#include <device_launch_parameters.h>
#define CUDA_KERNEL(grid, block)
#define <<<grid, block>>>
#endif

Linear::Linear(int in_f, int out_f) {

    in_features = in_f;
    out_features = out_f;

    W = new Tensor(in_f * out_f, true);
    b = new Tensor(out_f, true);

    float* h_W = new float[in_f * out_f];
    float* h_b = new float[out_f];

    srand(time(0));

    for (int i = 0; i < in_f * out_f; i++)
        h_W[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    for (int i = 0; i < out_f; i++)
        h_b[i] = 0.0f;

    W->fromHost(h_W);
    b->fromHost(h_b);

    delete[] h_W;
    delete[] h_b;

    input = nullptr;
}

Linear::~Linear() {
    delete W;
    delete b;
}

Tensor Linear::forward(Tensor& x, int batch_size) {

    // FIX Bug 2: store input as shared_ptr so it outlives this forward call
    // and stays valid for the backward_fn lambda captured across epochs.
    auto input_sptr = std::shared_ptr<Tensor>(&x, [](Tensor*) {});
    input = input_sptr.get();  // keep raw pointer for header compatibility

    Linear* self = this;

    Tensor out(batch_size * out_features, true);

    // Graph connection — shared_ptr keeps input alive as long as `out` lives
    out.prev.push_back(input_sptr);

    dim3 threads(16, 16);
    dim3 blocks(
        (out_features + threads.x - 1) / threads.x,
        (batch_size + threads.y - 1) / threads.y
    );

    cudaMemset(out.data, 0, batch_size * out_features * sizeof(float));

    matmul << <blocks, threads >> > (
        x.data, W->data, out.data,
        batch_size, out_features, in_features);

    addBias << <blocks, threads >> > (
        out.data, b->data, batch_size, out_features);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // FIX Bug 2: capture input_sptr (shared ownership) instead of raw input_ptr
    out.backward_fn =
        [input_sptr, self, batch_size](Tensor& grad_out) {

        Tensor grad_input(batch_size * self->in_features, true);

        int threads = 256;

        cudaMemset(self->W->grad, 0, self->W->size * sizeof(float));
        cudaMemset(self->b->grad, 0, self->b->size * sizeof(float));

        // dW
        int total_dW = self->in_features * self->out_features;
        int blocks_dW = (total_dW + threads - 1) / threads;
        matmul_backward_dW << <blocks_dW, threads >> > (
            input_sptr->data, grad_out.grad, self->W->grad,
            batch_size, self->out_features, self->in_features);

        // dX
        int total_dX = batch_size * self->in_features;
        int blocks_dX = (total_dX + threads - 1) / threads;
        matmul_backward_dX << <blocks_dX, threads >> > (
            grad_out.grad, self->W->data, grad_input.data,
            batch_size, self->out_features, self->in_features);

        // db
        int blocks_db = (self->out_features + threads - 1) / threads;
        reduce_sum_bias << <blocks_db, threads >> > (
            grad_out.grad, self->b->grad,
            batch_size, self->out_features);

        cudaDeviceSynchronize();

        if (input_sptr->grad == nullptr)
            cudaMalloc(&input_sptr->grad, grad_input.size * sizeof(float));

        cudaMemcpy(input_sptr->grad, grad_input.data,
            grad_input.size * sizeof(float),
            cudaMemcpyDeviceToDevice);

        input_sptr->backward();
        };

    return out;
}



//#include "linear_layer.h"
//#include "kernels.h"
//
//#include <cuda_runtime.h>
//#include <cstdio>
//#include <cstdlib>
//#include <ctime>
//
//#ifdef __INTELLISENSE__
//#define __CUDACC__
//#include <device_launch_parameters.h>
//#define CUDA_KERNEL(grid, block)
//#define <<<grid, block>>>
//#endif
//
//Linear::Linear(int in_f, int out_f) {
//
//    in_features = in_f;
//    out_features = out_f;
//
//    // Trainable parameters
//    W = new Tensor(in_f * out_f, true);
//    b = new Tensor(out_f, true);
//
//    float* h_W = new float[in_f * out_f];
//    float* h_b = new float[out_f];
//
//    srand(time(0));
//
//    for (int i = 0; i < in_f * out_f; i++) {
//
//        h_W[i] =
//            ((float)rand() / RAND_MAX - 0.5f)
//            * 0.1f;
//    }
//
//    for (int i = 0; i < out_f; i++) {
//        h_b[i] = 0.0f;
//    }
//
//    W->fromHost(h_W);
//    b->fromHost(h_b);
//
//    delete[] h_W;
//    delete[] h_b;
//
//    input = nullptr;
//}
//
//Linear::~Linear() {
//
//    delete W;
//    delete b;
//}
//
//Tensor Linear::forward(
//    Tensor& x,
//    int batch_size) {
//
//    // Store NON-OWNING input pointer
//    input = &x;
//
//    Tensor* input_ptr = input;
//    Linear* self = this;
//
//    // Output tensor
//    Tensor out(
//        batch_size * out_features,
//        true
//    );
//
//    // Graph connection
//    out.prev.push_back(
//        std::shared_ptr<Tensor>(
//            input_ptr,
//            [](Tensor*) {}
//        )
//    );
//
//    // Forward pass
//    dim3 threads(16, 16);
//
//    dim3 blocks(
//        (out_features + threads.x - 1)
//        / threads.x,
//
//        (batch_size + threads.y - 1)
//        / threads.y
//    );
//
//    cudaMemset(
//        out.data,
//        0,
//        batch_size *
//        out_features *
//        sizeof(float)
//    );
//
//    matmul << <blocks, threads >> > (
//        x.data,
//        W->data,
//        out.data,
//        batch_size,
//        out_features,
//        in_features
//        );
//
//    addBias << <blocks, threads >> > (
//        out.data,
//        b->data,
//        batch_size,
//        out_features
//        );
//
//    cudaDeviceSynchronize();
//
//    // CUDA debug
//    cudaError_t err = cudaGetLastError();
//
//    if (err != cudaSuccess) {
//
//        printf(
//            "CUDA Error: %s\n",
//            cudaGetErrorString(err)
//        );
//    }
//
//    // 🔥 AUTOGRAD BACKWARD FUNCTION
//    out.backward_fn =
//        [input_ptr, self, batch_size]
//        (Tensor& grad_out) {
//
//        // Input gradient
//        Tensor grad_input(
//            batch_size *
//            self->in_features,
//            true
//        );
//
//        int threads = 256;
//
//        // Zero parameter gradients
//        cudaMemset(
//            self->W->grad,
//            0,
//            self->W->size *
//            sizeof(float)
//        );
//
//        cudaMemset(
//            self->b->grad,
//            0,
//            self->b->size *
//            sizeof(float)
//        );
//
//        // =====================
//        // dW
//        // =====================
//
//        int total_dW =
//            self->in_features *
//            self->out_features;
//
//        int blocks_dW =
//            (total_dW + threads - 1)
//            / threads;
//
//        matmul_backward_dW << <
//            blocks_dW,
//            threads
//            >> > (
//                input_ptr->data,
//                grad_out.grad,
//                self->W->grad,
//                batch_size,
//                self->out_features,
//                self->in_features
//                );
//
//        // =====================
//        // dX
//        // =====================
//
//        int total_dX =
//            batch_size *
//            self->in_features;
//
//        int blocks_dX =
//            (total_dX + threads - 1)
//            / threads;
//
//        matmul_backward_dX << <
//            blocks_dX,
//            threads
//            >> > (
//                grad_out.grad,
//                self->W->data,
//                grad_input.data,
//                batch_size,
//                self->out_features,
//                self->in_features
//                );
//
//        // =====================
//        // db
//        // =====================
//
//        int blocks_db =
//            (self->out_features + threads - 1)
//            / threads;
//
//        reduce_sum_bias << <
//            blocks_db,
//            threads
//            >> > (
//                grad_out.grad,
//                self->b->grad,
//                batch_size,
//                self->out_features
//                );
//
//        cudaDeviceSynchronize();
//
//        // Persistent gradient storage
//        if (input_ptr->grad == nullptr) {
//
//            cudaMalloc(
//                &input_ptr->grad,
//                grad_input.size *
//                sizeof(float)
//            );
//        }
//
//        cudaMemcpy(
//            input_ptr->grad,
//            grad_input.data,
//            grad_input.size *
//            sizeof(float),
//            cudaMemcpyDeviceToDevice
//        );
//
//        // Recursive autograd
//        input_ptr->backward();
//        };
//
//    return out;
//}