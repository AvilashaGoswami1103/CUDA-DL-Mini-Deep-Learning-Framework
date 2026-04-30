#pragma once
#include "tensor.h"

class CrossEntropyLoss {    // class CrossEntropyLoss
public:
    // This class encapsulates the logic for computing cross-entropy loss and its gradient (backpropagation)
    float forward(Tensor& pred,
        Tensor& target,
        int batch_size,
        int num_classes);
    // pred: predictions from the model (softmax probabilities).
    // target: ground - truth labels(often one - hot encoded or class indices).
    // batch_size : number of samples in the batch.
    // num_classes : number of possible classes.

    Tensor backward(Tensor& pred,
        Tensor& target);
    // Parameters:
    // pred: predictions(same as forward).
    // target : ground - truth labels.
};
