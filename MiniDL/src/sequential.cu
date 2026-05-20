//#include "sequential.h"
//
//// Add layer
//void Sequential::add(Layer* layer) {    // Sequential::add means this function belongs to the Sequential class
//    layers.push_back(layer);
//    // layers is the std::vector<Layer*> member of the Sequential class
//    // push_back(layer) appends the given Layer* to the end of the vector
//}
//
//Tensor Sequential::forward(Tensor& x, int batch_size) {
//
//    Tensor* current = x;
//    /*Tensor* owned = nullptr;*/
//
//    //for (auto layer : layers) {
//    //    Tensor* next = new Tensor(layer->forward(*current, batch_size));
//    //    if (owned) delete owned;
//    //    owned = next;
//    //    current = owned;
//    //}
//
//    //Tensor result(*current);  // final copy
//    //if (owned) delete owned;
//    //return result;
//    for (auto layer : layers) {
//        current = layer->forward(current, batch_size);
//        // Each forward sets current.prev = shared_ptr to previous node
//        // shared_ptr keeps the whole chain alive as long as 'current' exists
//    }
//
//    return current;  // graph chain kept alive through shared_ptr chain
//}
//
//Tensor Sequential::backward(Tensor& grad, int batch_size) {
//    Tensor* current = &grad;
//    Tensor* owned = nullptr;
//
//    for (int i = layers.size() - 1; i >= 0; i--) {
//        Tensor* next = new Tensor(layers[i]->backward(*current, batch_size));
//        if (owned) delete owned;
//        owned = next;
//        current = owned;
//    }
//
//    Tensor result(*current);
//    if (owned) delete owned;
//    return result;
//    
//}

#include "sequential.h"
#include <utility>

void Sequential::add(Layer* layer) {
    layers.push_back(layer);
}

Tensor Sequential::forward(Tensor& x, int batch_size) {
    auto prev_sptr = std::shared_ptr<Tensor>(&x, [](Tensor*) {});
    std::shared_ptr<Tensor> current_sptr;

    // Keep ALL intermediates alive — each one is referenced by the next
    // layer's backward_fn via no-op ptr, so none can be destroyed early
    std::vector<std::shared_ptr<Tensor>> all_nodes;
    all_nodes.push_back(prev_sptr);

    for (auto layer : layers) {
        Tensor out = layer->forward(*prev_sptr, batch_size);
        current_sptr = std::make_shared<Tensor>(std::move(out));
        all_nodes.push_back(current_sptr);
        prev_sptr = current_sptr;
    }

    // Attach all intermediates to the output's prev so they stay alive
    // as long as the returned Tensor lives (which in main.cu spans backward)
    for (auto& node : all_nodes) {
        current_sptr->prev.push_back(node);
    }

    return std::move(*current_sptr);
}
//Tensor Sequential::backward(Tensor& grad, int batch_size) {
//    Tensor current = grad;
//
//    for (int i = layers.size() - 1; i >= 0; i--) {
//        Tensor next = layers[i]->backward(current, batch_size);
//        current = std::move(next);
//    }
//
//    return current;
//}