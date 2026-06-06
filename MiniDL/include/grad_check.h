#pragma once
#include "tensor.h"
#include <functional>
#include <cstdio>
#include <cmath>

// Checks gradient of one parameter numerically vs analytically.
// model_fn: runs forward+backward and returns loss as float
// param: the parameter tensor to check
// eps: perturbation size (1e-4 is standard)
// tol: pass/fail tolerance (1e-2 is reasonable for float32)
void grad_check(
    std::function<float()> model_fn,
    Tensor* param,
    float eps = 1e-4f,
    float tol = 1e-2f);
