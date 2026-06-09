#include "tensor.h"
#include "layer.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <vector> // A sequence container that stores elements in contiguous memory.
#include <unordered_set> // An unordered associative container that stores unique elements using a hash table.

// constructor for Tensor class
Tensor::Tensor(int size, bool requires_grad) {
    this->size = size;  // stores the number of elements
    this->requires_grad = requires_grad;    // stores whether gradients should be tracked

    CUDA_CHECK(cudaMalloc(&data, size * sizeof(float)));
    CUDA_CHECK(cudaMemset(data, 0, size * sizeof(float)));
    /*cudaMalloc allocates GPU memory for the tensor’s data.
    cudaMemset initializes that memory to zero.
    CUDA_CHECK is a macro that likely checks for CUDA errors after each call.*/

    // same for gradient is requires_grad is set true
    if (requires_grad) {
        CUDA_CHECK(cudaMalloc(&grad, size * sizeof(float)));
        CUDA_CHECK(cudaMemset(grad, 0, size * sizeof(float)));
    }

    else {
        grad = nullptr;
    }
}

// Copy constructor — copies GPU data only, NO graph transfer.
// A copy is a new value-only leaf node.
Tensor::Tensor(const Tensor& other) {
    size = other.size;
    requires_grad = other.requires_grad;

    cudaMalloc(&data, size * sizeof(float));
    cudaMemcpy(data, other.data, size * sizeof(float), cudaMemcpyDeviceToDevice);
    /*Allocates fresh GPU memory for the new tensor’s data.
    Copies the contents from other.data using cudaMemcpy(device - to - device transfer).
    This ensures the new tensor has its own independent memory.*/

    if (requires_grad && other.grad != nullptr) {
        cudaMalloc(&grad, size * sizeof(float));
        cudaMemcpy(grad, other.grad, size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    // If gradients are required and the source tensor has a gradient buffer, allocate and copy it too.
    else {
        grad = nullptr;
    }
    // backward_fn and prev intentionally NOT copied — copy is a leaf node
}
//This copy constructor ensures that when you duplicate a Tensor, you get a new independent 
//tensor with its own GPU memory.It copies the data and gradients but
//deliberately does not copy the computational graph, so the new tensor becomes a fresh leaf node.

// Copy assignment — same rule: values only, no graph
Tensor& Tensor::operator=(const Tensor& other) {
    // The copy assignment operator (operator=) defines how one existing object should be assigned the values of another object.
    if (this != &other) {   // Prevents problems if you accidentally assign an object to itself (a = a;).
        if (data) cudaFree(data);
        if (grad) cudaFree(grad);

        size = other.size;
        requires_grad = other.requires_grad;

        cudaMalloc(&data, size * sizeof(float));
        cudaMemcpy(data, other.data, size * sizeof(float), cudaMemcpyDeviceToDevice);

        if (requires_grad && other.grad != nullptr) {
            cudaMalloc(&grad, size * sizeof(float));
            cudaMemcpy(grad, other.grad, size * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        else {
            grad = nullptr;
        }
        // backward_fn and prev intentionally NOT copied
    }
    return *this;
}

// Move constructor — transfers EVERYTHING including the graph
// A move constructor defines how an object should be created by taking ownership of another object’s resources instead of copying them.
Tensor::Tensor(Tensor&& other) noexcept {
    //transfer raw data pointers
    data = other.data;
    grad = other.grad;
    // copy metadata
    size = other.size;
    requires_grad = other.requires_grad;
    // transfer graph connections
    backward_fn = std::move(other.backward_fn);
    prev = std::move(other.prev);
    // nullify the source
    other.data = nullptr;
    other.grad = nullptr;
}

// Move assignment — transfers EVERYTHING including the graph
// The move assignment operator (operator=(Tensor&& other)) defines how an already existing object should take ownership of another object’s resources when assigned with std::move
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (data) cudaFree(data);
        if (grad) cudaFree(grad);

        data = other.data;
        grad = other.grad;
        size = other.size;
        requires_grad = other.requires_grad;

        backward_fn = std::move(other.backward_fn);
        prev = std::move(other.prev);

        other.data = nullptr;
        other.grad = nullptr;
    }
    return *this;
}

void Tensor::fromHost(float* h_data) {
    cudaMemcpy(data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);
}

void Tensor::toHost(float* h_data) {
    cudaMemcpy(h_data, data, size * sizeof(float), cudaMemcpyDeviceToHost);
}

// The purpose of zero_grad() is to reset the gradient buffer of the tensor back to zero.
void Tensor::zero_grad() {
    if (requires_grad && grad != nullptr) {
        cudaMemset(grad, 0, size * sizeof(float));
    }
}

// build_topo recursively traverses the computational graph of tensors and produces a topological sort — an ordering where each tensor appears after all of its dependencies.This ordering is crucial for backpropagation, ensuring gradients are computed in the right sequence.
// This ordering is crucial for backpropagation, ensuring gradients are computed in the right sequence.
// topological sort helper function for tensors in a computational graph
static void build_topo(Tensor* t,
    std::vector<Tensor*>& order,
    std::unordered_set<Tensor*>& visited) {

    if (visited.count(t)) return;
    visited.insert(t);

    for (auto& p : t->prev)
        build_topo(p.get(), order, visited);

    /*Each tensor keeps track of its predecessors(prev) in the computational graph.
    For every predecessor, recursively call build_topo.
    This ensures we process dependencies before the current tensor.*/

    order.push_back(t);
    // After all predecessors are processed, add the current tensor to the order list.
}

void Tensor::backward() {
    // Seed gradient = 1 for the root (loss scalar)
    if (grad == nullptr) {
        cudaMalloc(&grad, size * sizeof(float));
    }
    float one = 1.0f;
    cudaMemcpy(grad, &one, sizeof(float), cudaMemcpyHostToDevice);

    /*The starting point of backpropagation is the loss scalar(usually a single value).
    Its gradient with respect to itself is defined as 1.
    This seeds the chain rule : every other gradient flows backward from this root.*/

    // Build topological order of the graph
    std::vector<Tensor*> order;
    std::unordered_set<Tensor*> visited;
    build_topo(this, order, visited);

    // Traverse in reverse — from loss back to inputs
    for (int i = (int)order.size() - 1; i >= 0; i--) {
        Tensor* t = order[i];
        if (t->backward_fn)
            t->backward_fn(*t);
    }
}

Tensor::~Tensor() {
    if (data) cudaFree(data);
    if (grad) cudaFree(grad);
}

void Tensor::free_graph() {
    backward_fn = nullptr;
    prev.clear();
}
