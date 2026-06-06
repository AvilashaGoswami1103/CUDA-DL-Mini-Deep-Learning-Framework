#include "grad_check.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

void grad_check(
    std::function<float()> model_fn,
    Tensor* param,
    float eps,
    float tol) {

    int n = param->size;

    // Copy current param values to CPU
    std::vector<float> h_param(n);
    cudaMemcpy(h_param.data(), param->data,
        n * sizeof(float), cudaMemcpyDeviceToHost);

    // Get analytic gradient by running forward+backward once
    float analytic_loss = model_fn();
    std::vector<float> h_grad(n);
    cudaMemcpy(h_grad.data(), param->grad,
        n * sizeof(float), cudaMemcpyDeviceToHost);

    // Check each element numerically
    int errors = 0;
    int checked = 0;
    int max_check = 20;  // only check first 20 elements — enough to verify

    printf("\nGradient check for param (size=%d), checking %d elements:\n",
        n, max_check);

    for (int i = 0; i < n && checked < max_check; i++, checked++) {

        // Perturb param[i] + eps
        std::vector<float> h_plus = h_param;
        h_plus[i] += eps;
        cudaMemcpy(param->data, h_plus.data(),
            n * sizeof(float), cudaMemcpyHostToDevice);
        float loss_plus = model_fn();

        // Perturb param[i] - eps
        std::vector<float> h_minus = h_param;
        h_minus[i] -= eps;
        cudaMemcpy(param->data, h_minus.data(),
            n * sizeof(float), cudaMemcpyHostToDevice);
        float loss_minus = model_fn();

        // Restore original value
        cudaMemcpy(param->data, h_param.data(),
            n * sizeof(float), cudaMemcpyHostToDevice);

        // Numerical gradient
        float numerical = (loss_plus - loss_minus) / (2.0f * eps);
        float analytic = h_grad[i];

        // Relative error
        float diff = fabsf(numerical - analytic);
        float denom = fmaxf(fabsf(numerical) + fabsf(analytic), 1e-8f);
        float rel_error = diff / denom;

        bool pass = (rel_error < tol);
        if (!pass) errors++;

        printf("  [%3d] numerical=%.6f  analytic=%.6f  "
            "rel_err=%.6f  %s\n",
            i, numerical, analytic, rel_error,
            pass ? "OK" : "FAIL");
    }

    printf("Result: %d/%d passed (tolerance=%.4f)\n\n",
        checked - errors, checked, tol);
}