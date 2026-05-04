#include "sequential.h"

// Add layer
void Sequential::add(Layer* layer) {    // Sequential::add means this function belongs to the Sequential class
    layers.push_back(layer);
    // layers is the std::vector<Layer*> member of the Sequential class
    // push_back(layer) appends the given Layer* to the end of the vector
}

Tensor Sequential::forward(Tensor& x, int batch_size) {

    Tensor* current = &x;
    // Makes a copy of the input tensor x into a new variable out.
    // This out will be updated step by step as it flows through each layer

    Tensor out(1, false);

    // auto layer means each element is treated as a pointer to a Layer
    for (auto layer : layers) {
        out = layer->forward(*current, batch_size);
        //calls forward method of the current layer
        current = &out;
    }

    return out;
    // Returns the final output tensor after all layers have processed it.
}

Tensor Sequential::backward(Tensor& grad, int batch_size) {

    Tensor* current = &grad;

    Tensor d(1, false);

    for (int i = layers.size() - 1; i >= 0; i--) {
        d = layers[i]->backward(*current, batch_size);
        current = &d;
    }

    return d;
}