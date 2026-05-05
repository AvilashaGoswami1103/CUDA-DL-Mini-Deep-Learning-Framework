#include "sequential.h"

// Add layer
void Sequential::add(Layer* layer) {    // Sequential::add means this function belongs to the Sequential class
    layers.push_back(layer);
    // layers is the std::vector<Layer*> member of the Sequential class
    // push_back(layer) appends the given Layer* to the end of the vector
}

Tensor Sequential::forward(Tensor& x, int batch_size) {

    Tensor out = x;   // start from input

    for (auto layer : layers) {
        out = layer->forward(out, batch_size);
    }

    return out;
}

Tensor Sequential::backward(Tensor& grad, int batch_size) {

    Tensor d = grad;

    for (int i = layers.size() - 1; i >= 0; i--) {
        d = layers[i]->backward(d, batch_size);
    }

    return d;
}