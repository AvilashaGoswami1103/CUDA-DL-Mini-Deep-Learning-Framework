#include "softmax.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(float* input, float* output, int batch_size, int num_classes) {
    // input: pointer to input data (flattened 2D array: batch × classes).
    // output: pointer to output data(same shape).
    // batch_size : number of rows(samples).
    // num_classes : number of columns(class scores per sample)

    int row = blockIdx.x;   // each CUDA block handles 1 row(sample) 

    if (row < batch_size) {

        float sum = 0.0f;

        // exponentials
        for (int j = 0; j < num_classes; j++) {

            float val = expf(input[row * num_classes + j]);

            output[row * num_classes + j] = val;

            sum += val;
        }
        // Iterates over all classes in the current row.
        // input[row * num_classes + j]: accesses the j - th class score for this row.
        // expf(...) : computes exponential of that score.
        // Stores the exponential in output temporarily.
        // Adds it to sum(needed for normalization).

        // normalize
        for (int j = 0; j < num_classes; j++) {
            output[row * num_classes + j] /= sum;
        }
        // Divides each exponential by the total sum → softmax normalization.
        // After this, output[row][j] contains the softmax probability for class j
    }
}

Tensor Softmax::forward(Tensor& x,
    int batch_size,
    int num_classes) {      // member function of Softmax class

    Tensor out(x.size, false);  // Creates an output tensor out with the same size as x

    softmax_kernel << <batch_size, 1 >> > (
        x.data,
        out.data,
        batch_size,
        num_classes
        );
    // launches the CUDA kernel
    // <<<batch_size, 1>>>: grid has batch_size blocks, each with 1 thread

    cudaDeviceSynchronize();

    output = new Tensor(out.size, false);   // allocates a new tensor output

    cudaMemcpy(output->data,
        out.data,
        out.size * sizeof(float),
        cudaMemcpyDeviceToDevice);
    // copies the computed softmax results from out to output
    // Finally, the output tensor contains the softmax probabilities for each row (sample) in the batch

    return out;
}