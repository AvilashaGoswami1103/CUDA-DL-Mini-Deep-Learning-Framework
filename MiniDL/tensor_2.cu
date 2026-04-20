// Tensor with Gradients
#include <iostream>
#include <vector>   // for dynamic arrays
#include <cuda_runtime.h>

using namespace std;

class Tensor {
public:
    float* data;     // data is a pointer to GPU memory holding the tensor’s values.
    float* grad;     // grad is a pointer to GPU memory holding gradients (used in machine learning backprop)
    int size;        // number of elements

    bool requires_grad;     // flag indicating whether gradients should be tracked

    Tensor(int size, bool requires_grad = false) {  // Constructor: takes the tensor size and an optional flag
        // store values in object
        this->size = size;
        this->requires_grad = requires_grad;

        cudaMalloc(&data, size * sizeof(float));

        if (requires_grad) {
            cudaMalloc(&grad, size * sizeof(float));    // Allocates GPU memory for data using cudaMalloc
            cudaMemset(grad, 0, size * sizeof(float));  //  allocate GPU memory for grad and initialize it to zeros with cudaMemset
        }
        else {
            grad = nullptr; // otherwise, no gradient memory allocated
        }
    }

    void fromHost(float* h_data) {
        cudaMemcpy(data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);
    }

    void toHost(float* h_data) {
        cudaMemcpy(h_data, data, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void zero_grad() {
        if (requires_grad) {
            cudaMemset(grad, 0, size * sizeof(float));
        }
    }
    // Resets the gradient buffer to zero (only if gradients are being tracked)
    // This is common in training loops to clear old gradients before computing new ones.

    ~Tensor() {
        cudaFree(data);
        if (grad) cudaFree(grad);
    }
    // Destructor: automatically called when the object is destroyed.
    // Free GPU memory for data
    // if grad was allocated, frees that too
};

// Your Tensor class has a grad buffer on the GPU. This is where the gradient values for each element of the tensor are stored.
// When you run training, you compute gradients of the loss with respect to each tensor.Those values are written into grad.
// The flag requires_grad decides whether a tensor should track gradients(parameters need them, but maybe input data doesn’t).
// The method zero_grad() clears out old gradients before the next training step, so they don’t accumulate incorrectly.
// After gradients are computed, an optimizer(like SGD or Adam) would use them to adjust the tensor’s data values(the actual parameters).

// Minimal Test
int main() {
    int N = 5;

    float h_data[5] = { 1,2,3,4,5 };

    Tensor t(N, true);  // Tensor object t constructed of size 5, requires_grad is enabled

    t.fromHost(h_data); //copy into GPU memory

    t.zero_grad();  //clears gradient buffer before computing new gradients in training loops

    float out[5];   //allocates a CPU array of size 5
    t.toHost(out);  //copies tensor's GPU data back into CPU array

    for (int i = 0; i < N; i++)
        cout << out[i] << " ";

    cout << endl;

    return 0;
}