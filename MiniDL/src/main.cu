#include <iostream>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>

#include "tensor.h"
#include "linear_layer.h"
#include "optimizer.h"
#include "relu.h"
#include "softmax.h"
#include "cross_entropy.h"
#include "sequential.h"

using namespace std;

int main() {

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    cout << "CUDA Devices: " << deviceCount << endl;

    // Network dimensions
    int batch = 2;
    int in_f = 3;
    int hidden_f = 4;
    int out_f = 2;

    // Input data
    float h_input[] = { 1, 2, 3, 4, 5, 6 };
    Tensor x(batch * in_f, false);
    x.fromHost(h_input);

    // Build model
    Linear layer1(in_f, hidden_f);
    ReLU   relu;
    Linear layer2(hidden_f, out_f);

    Sequential model;
    model.add(&layer1);
    model.add(&relu);
    model.add(&layer2);

    Softmax          softmax(out_f);
    CrossEntropyLoss loss_fn;
    SGD              optimizer(0.01f);

    optimizer.add_param(layer1.W);
    optimizer.add_param(layer1.b);
    optimizer.add_param(layer2.W);
    optimizer.add_param(layer2.b);

    // Fixed target labels (one-hot)
    float h_target[] = { 1, 0,   // sample 0: class 0
                         0, 1 }; // sample 1: class 1

    // Print input
    float temp[6];
    x.toHost(temp);
    cout << "Input: ";
    for (int i = 0; i < 6; i++) cout << temp[i] << " ";
    cout << "\n" << endl;

    int epochs = 20;

    for (int epoch = 0; epoch < epochs; epoch++) {

        // --- 1. Zero gradients ---
        optimizer.zero_grad();

        // --- 2. Forward pass ---
        // All tensors declared here so they stay alive through backward()
        Tensor logits = model.forward(x, batch);
        Tensor out = softmax.forward(logits, batch);

        Tensor target(batch * out_f, false);
        target.fromHost(h_target);

        Tensor loss = loss_fn.forward(out, target, batch, out_f);

        // --- 3. Backward pass ---
        // loss already has grad=1 seeded inside CrossEntropyLoss::forward
        // backward_fn propagates: loss -> softmax out -> logits -> layer2 -> relu -> layer1
        loss.backward();
        loss.free_graph();

        // --- 4. Optimizer step ---
        optimizer.step();

        // --- 5. Print results ---
        float h_loss;
        loss.toHost(&h_loss);

        float h_out[4];
        out.toHost(h_out);

        cout << "Epoch " << epoch
            << "  Loss: " << h_loss
            << "  Output: ";
        for (int i = 0; i < batch * out_f; i++)
            cout << h_out[i] << " ";
        cout << endl;

        this_thread::sleep_for(chrono::milliseconds(100));
    }

    return 0;
}
