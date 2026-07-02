#pragma once    
#include "tensor.h"
#include "layer.h"

class Softmax : public Layer {  // declare a softmax class that inherits from layer
public:
    int num_classes;

    Softmax(int num_classes) : num_classes(num_classes) {}  // initializes the layer with the number of classes.
    // stores this value for use in forward computation
    Tensor forward(Tensor& x, int batch_size);
    // Declares the forward pass of Softmax.
    // this is apply softmax across each row of the input (each sample of the batch) -> Produces probabilities for each class that sum to 1.
    //Tensor backward(Tensor& grad, int batch_size = 0);
};