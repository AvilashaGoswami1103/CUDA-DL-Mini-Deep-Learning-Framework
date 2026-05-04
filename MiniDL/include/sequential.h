// declares a sequential container
#pragma once

#include <vector>   // C++ Standard Library vector container
// std:vector -> dynamic array that can grow or shrink in size, used to store collections of objects
#include "layer.h"

class Sequential {
public:
    std::vector<Layer*> layers;
    // Stores all layers dynamically: [Linear → ReLU → Linear]
    // This is the core data structure holding the ordered list of layers in the network.
    // Using pointers allows polymorphism(different types of layers can be stored in the same vector).
    void add(Layer* layer);

    // takes a pointer to a Layer and adds it to the layers vector

    Tensor forward(Tensor& x, int batch_size);
    // forward method: takes a reference to Tensor x, input data
    // An integer batch_size, which specifies how many samples are processed at once

    Tensor backward(Tensor& grad, int batch_size);
    // It takes:
    // A reference to a Tensor(grad), which represents the gradient of the loss with respect to the output.
    // An integer batch_size
};
