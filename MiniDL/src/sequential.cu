#include "sequential.h"
#include <utility>
#include <memory>

void Sequential::add(Layer* layer) {
    layers.push_back(layer);
}

Tensor Sequential::forward(Tensor& x, int batch_size) {

    // Wrap input with no-op deleter — x is owned by main, not us
    auto prev_sptr = std::shared_ptr<Tensor>(&x, [](Tensor*) {});
    std::shared_ptr<Tensor> current_sptr;

    // Keep ALL intermediate tensors alive in this vector.
    // Each layer's backward_fn holds a no-op ptr into one of these nodes,
    // so they must stay alive until after loss.backward() completes.
    std::vector<std::shared_ptr<Tensor>> all_nodes;
    all_nodes.push_back(prev_sptr);

    for (auto layer : layers) {
        Tensor out = layer->forward(*prev_sptr, batch_size);
        current_sptr = std::make_shared<Tensor>(std::move(out));
        all_nodes.push_back(current_sptr);
        prev_sptr = current_sptr;
    }

    // Pin all intermediates onto the output's prev vector.
    // This means as long as the returned Tensor (logits in main) is alive,
    // every intermediate node is alive too — covering the full backward pass.
    for (auto& node : all_nodes) {
        current_sptr->prev.push_back(node);
    }

    return std::move(*current_sptr);
}
