#pragma once
#include "tensor.h"
// SGD will need to update parameters stored as Tensor objects

// Stochastic gradient descent
class SGD {
public:     // can be accessed from outside the class
    float lr;   //learning rate 
    // lr -> controls the step size of the parameter update

    SGD(float lr);  // constructor declaration
    // to create a SGD object with a specific learning rate

    void step(Tensor* param);
    // method step, takes a pointer to a Tensor
    // Tensor -> performs actual update subtracting lr × gradient from the parameter’s values.
};
