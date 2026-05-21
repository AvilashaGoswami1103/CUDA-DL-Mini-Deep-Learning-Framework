#pragma once
#include "tensor.h"
#include <vector>

class SGD {
public:
    float lr;
    std::vector<Tensor*> parameters;  // ADD THIS

    SGD(float lr);
    void add_param(Tensor* param);    // ADD THIS
    void zero_grad();                 // ADD THIS
    void step(Tensor* param);

    void step();
    void step(Tensor* param);
};
