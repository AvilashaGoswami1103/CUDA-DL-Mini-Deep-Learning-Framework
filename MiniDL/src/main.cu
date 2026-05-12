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

    // 🔥 Check GPU
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    cout << "CUDA Devices: " << deviceCount << endl;

    // 🔥 FORCE GPU usage (memory test)
    float* d_test;
    cudaMalloc(&d_test, 100000000 * sizeof(float)); // ~400MB
    cudaDeviceSynchronize();

    // 🔧 Model dimensions
    int batch = 2;
    int in_f = 3;
    int hidden_f = 4;
    int out_f = 2;

    // 🔧 Input
    float h_input[] = {
        1, 2, 3,
        4, 5, 6
    };

    Tensor x(batch * in_f, false);
    x.fromHost(h_input);

    // 🔧 Model (MLP)
    Linear layer1(in_f, hidden_f);
    ReLU relu;
    Linear layer2(hidden_f, out_f);

    Sequential model;
    model.add(&layer1);
    model.add(&relu);
    model.add(&layer2);

    Softmax softmax(out_f);
    CrossEntropyLoss loss_fn;

    // 🔍 Debug input
    float temp[6];
    x.toHost(temp);

    cout << "Input: ";
    for (int i = 0; i < 6; i++) cout << temp[i] << " ";
    cout << endl;

    // 🔧 Training setup
    int epochs = 20;
    SGD optimizer(0.01f);

    for (int epoch = 0; epoch < epochs; epoch++) {

        // 🔥 RESET gradients every epoch
        layer1.W->zero_grad();
        layer1.b->zero_grad();
        layer2.W->zero_grad();
        layer2.b->zero_grad();

        // 🔹 Forward
        Tensor logits = model.forward(x, batch);

        Tensor out = softmax.forward(logits, batch);

        float* h_out = new float[batch * out_f];
        out.toHost(h_out);

        // 🔹 Target (one-hot)
        float h_target[] = {
            1, 0,
            0, 1
        };

        Tensor target(batch * out_f, false);
        target.fromHost(h_target);

        // 🔹 Loss
        float loss = loss_fn.forward(out, target, batch, out_f);

        // 🔥 FIXED backward call
        Tensor d_out = loss_fn.backward(out, target, batch);

        // 🔹 Backward
        Tensor dX = model.backward(d_out, batch);
        // model

        // 🔹 Update
        optimizer.step(layer1.W);
        optimizer.step(layer1.b);
        optimizer.step(layer2.W);
        optimizer.step(layer2.b);

        // 🔍 Print
        cout << "Epoch " << epoch << " Loss: " << loss << " Output: ";
        for (int i = 0; i < batch * out_f; i++)
            cout << h_out[i] << " ";
        cout << endl;

        delete[] h_out;

        // 🔥 Slow down so GPU usage is visible
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // 🔥 Free test memory
    cudaFree(d_test);

    return 0;
}