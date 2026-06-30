#include "sequential.h"
#include <utility>
#include <memory>

void Sequential::add(Layer* layer) {
    layers.push_back(layer);    // Adds a new layer to the internal list (layers).
}   // allows to build a stack of layers one by one

Tensor Sequential::forward(Tensor& x, int batch_size) { // defines how input flows through all layers in sequence.

    // Wrap input with no-op deleter — x is owned by main, not us
    auto prev_sptr = std::shared_ptr<Tensor>(&x, [](Tensor*) {});
    // Wraps the input tensor in a non-owning smart pointer (no-op deleter).
    std::shared_ptr<Tensor> current_sptr;
    /*prev_sptr points to the current input for each layer.
    current_sptr will hold the output of each layer.*/

    // Keep ALL intermediate tensors alive in this vector.
    // Each layer's backward_fn holds a no-op ptr into one of these nodes,
    // so they must stay alive until after loss.backward() completes.
    std::vector<std::shared_ptr<Tensor>> all_nodes;
    all_nodes.push_back(prev_sptr);
    // all_nodes → keeps all intermediate tensors alive for autograd.

    // iterates through each layer
    for (auto layer : layers) {
        Tensor out = layer->forward(*prev_sptr, batch_size);    // calls its forward() with the current input
        current_sptr = std::make_shared<Tensor>(std::move(out));    // wraps the input in a shared pointer
        all_nodes.push_back(current_sptr);  // stores it in all_nodes
        prev_sptr = current_sptr;   // Sets prev_sptr to this output for the next layer.
    }

    // Pin all intermediates onto the output's prev vector.
    // This means as long as the returned Tensor (logits in main) is alive,
    // every intermediate node is alive too — covering the full backward pass.
    for (auto& node : all_nodes) {
        current_sptr->prev.push_back(node);
    }

    return std::move(*current_sptr);
    // Returns the final output tensor (moved out of the shared pointer).
}
