#include "sequential.h"

// Add layer
void Sequential::add(Layer* layer) {    // Sequential::add means this function belongs to the Sequential class
    layers.push_back(layer);
    // layers is the std::vector<Layer*> member of the Sequential class
    // push_back(layer) appends the given Layer* to the end of the vector
}

Tensor Sequential::forward(Tensor& x, int batch_size) {

    Tensor* current = x;
    /*Tensor* owned = nullptr;*/

    //for (auto layer : layers) {
    //    Tensor* next = new Tensor(layer->forward(*current, batch_size));
    //    if (owned) delete owned;
    //    owned = next;
    //    current = owned;
    //}

    //Tensor result(*current);  // final copy
    //if (owned) delete owned;
    //return result;
    for (auto layer : layers) {
        current = layer->forward(current, batch_size);
        // Each forward sets current.prev = shared_ptr to previous node
        // shared_ptr keeps the whole chain alive as long as 'current' exists
    }

    return current;  // graph chain kept alive through shared_ptr chain
}

Tensor Sequential::backward(Tensor& grad, int batch_size) {
    Tensor* current = &grad;
    Tensor* owned = nullptr;

    for (int i = layers.size() - 1; i >= 0; i--) {
        Tensor* next = new Tensor(layers[i]->backward(*current, batch_size));
        if (owned) delete owned;
        owned = next;
        current = owned;
    }

    Tensor result(*current);
    if (owned) delete owned;
    return result;
    
}