#pragma once
#include "tensor.h"
//include tensor.h file

class MSELoss {     // class definition for mean squared error, loss function
public:
    float forward(Tensor& pred, Tensor& target);
    // declares member function "forward":
    // Tensor& pred: takes a reference to the predicted values (model outputs)
    // Tensor& target: a reference to the ground-truth values(labels)
    // return float: represents the computed mean squared error value
    Tensor backward(Tensor& pred, Tensor& target);
    // declares another member function named function named backward
    // takes predicted and target tensors as input
    // returns a Tensor, represents the gradient of loss wrt to the predictions

    // MSE: (pred - target) * (2 / N)
};
