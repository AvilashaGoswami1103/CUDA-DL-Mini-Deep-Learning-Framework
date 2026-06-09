#include <iostream>
#include <thread>   //provides support for multithreading - parallel programming
#include <chrono>   // Provides time utilities for measuring durations and working with clocks.
#include <cuda_runtime.h>
#include <iomanip>  // provides manipulators for formatted input/output streams. 
// E.g: field width, decimal precision, floating-point formatting, fill character for padding

#include "tensor.h"
#include "linear_layer.h"
#include "optimizer.h"
#include "relu.h"
#include "adam.h"
#include "softmax.h"
#include "cross_entropy.h"
#include "sequential.h"
#include "dropout.h"
#include "autograd_context.h"
#include "batchnorm.h"
#include "conv2d.h"
#include "data.h"
#include "checkpoint.h"
#include "grad_check.h"

using namespace std;

int main() {

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    cout << "CUDA Devices: " << deviceCount << endl;

    // Network dimensions
    int batch = 32;
    int in_f = 3;
    int hidden_f = 8;
    int out_f = 2;

    // Input data
    Tensor x(batch * in_f, false);
    x.fromHost(train_data);

    // Build model
    Linear    layer1(in_f, hidden_f);
    BatchNorm bn1(hidden_f);
    ReLU      relu;
    Dropout   dropout(0.3f);
    Linear    layer2(hidden_f, out_f);

    Sequential model;
    model.add(&layer1);
    model.add(&bn1);
    model.add(&relu);
    model.add(&dropout);
    model.add(&layer2);

    Softmax          softmax(out_f);
    CrossEntropyLoss loss_fn;
    Adam             optimizer(0.001f);

    optimizer.add_param(layer1.W);
    optimizer.add_param(layer1.b);
    optimizer.add_param(bn1.gamma);
    optimizer.add_param(bn1.beta);
    optimizer.add_param(layer2.W);
    optimizer.add_param(layer2.b);

    bn1.set_training(true);
    dropout.set_training(true);

    int epochs = 500;

    cout << "Training on " << batch << " samples for "
        << epochs << " epochs\n" << endl;

    // Target labels — declared outside loop, never changes
    Tensor target(batch * out_f, false);
    target.fromHost(train_labels);

    // FIX: correct buffer size — batch * out_f = 32 * 2 = 64
    float h_out[64];

    for (int epoch = 0; epoch < epochs; epoch++) {

        // --- 1. Zero gradients ---
        optimizer.zero_grad();

        // --- 2. Forward pass ---
        Tensor logits = model.forward(x, batch);
        Tensor out = softmax.forward(logits, batch);
        Tensor loss = loss_fn.forward(out, target, batch, out_f);

        // --- 3. Backward pass ---
        loss.backward();
        loss.free_graph();

        // --- 4. Gradient check — runs ONLY on epoch 0, then never again ---
        if (epoch == 0) {
            auto check_fn = [&]() -> float {
                optimizer.zero_grad();
                Tensor lg = model.forward(x, batch);
                Tensor ot = softmax.forward(lg, batch);
                Tensor ls = loss_fn.forward(ot, target, batch, out_f);
                ls.backward();
                float h;
                ls.toHost(&h);
                ls.free_graph();
                return h;
                };
            grad_check(check_fn, layer1.W);
            grad_check(check_fn, layer2.W);
        }

        // --- 5. Optimizer step ---
        optimizer.step();

        // --- 6. Print every 10 epochs ---
        if (epoch % 10 == 0 || epoch == epochs - 1) {

            float h_loss;
            loss.toHost(&h_loss);
            out.toHost(h_out);

            // Accuracy
            int correct = 0;
            for (int i = 0; i < batch; i++) {
                int pred = (h_out[i * 2] > h_out[i * 2 + 1]) ? 0 : 1;
                int label = (train_labels[i * 2] > train_labels[i * 2 + 1]) ? 0 : 1;
                if (pred == label) correct++;
            }
            float accuracy = (float)correct / batch * 100.0f;

            // Per-class average confidence
            float avg_conf_class0 = 0.0f;  // avg p(class0) for class-0 samples
            float avg_conf_class1 = 0.0f;  // avg p(class1) for class-1 samples
            for (int i = 0; i < batch; i++) {
                int label = (train_labels[i * 2] > train_labels[i * 2 + 1]) ? 0 : 1;
                if (label == 0) avg_conf_class0 += h_out[i * 2];
                else            avg_conf_class1 += h_out[i * 2 + 1];
            }
            avg_conf_class0 /= (batch / 2);
            avg_conf_class1 /= (batch / 2);

            cout << "Epoch " << setw(3) << epoch
                << "  Loss: " << h_loss
                << "  Acc: " << accuracy << "%"
                << "  Conf0: " << avg_conf_class0
                << "  Conf1: " << avg_conf_class1
                << endl;
        }
    }
    // Save after training
    std::vector<Tensor*> params = {
        layer1.W, layer1.b,
        bn1.gamma, bn1.beta,
        layer2.W, layer2.b
    };
    save_checkpoint(params, "minidl_weights.bin");

    // To load instead of training (add this before the loop with an if/else):
    // load_checkpoint(params, "minidl_weights.bin");

    // -------------------------------------------
    // INFERENCE
    // -------------------------------------------
    cout << "\n--- Inference (dropout off, grad off) ---" << endl;

    bn1.set_training(false);
    dropout.set_training(false);
    AutogradContext::set_grad_enabled(false);

    Tensor infer_logits = model.forward(x, batch);
    Tensor infer_out = softmax.forward(infer_logits, batch);

    float h_infer[64];
    infer_out.toHost(h_infer);

    // Accuracy
    int infer_correct = 0;
    for (int i = 0; i < batch; i++) {
        int pred = (h_infer[i * 2] > h_infer[i * 2 + 1]) ? 0 : 1;
        int label = (train_labels[i * 2] > train_labels[i * 2 + 1]) ? 0 : 1;
        if (pred == label) infer_correct++;
    }

    // Per-sample predictions
    cout << "\nSample predictions (pred | label | confidence):" << endl;
    for (int i = 0; i < batch; i++) {
        int pred = (h_infer[i * 2] > h_infer[i * 2 + 1]) ? 0 : 1;
        int label = (train_labels[i * 2] > train_labels[i * 2 + 1]) ? 0 : 1;
        float conf = (pred == 0) ? h_infer[i * 2] : h_infer[i * 2 + 1];
        cout << "  Sample " << setw(2) << i
            << "  Pred: " << pred
            << "  Label: " << label
            << "  Conf: " << conf
            << (pred == label ? "" : "  ← WRONG")
            << endl;
    }

    cout << "\nFinal Accuracy: "
        << (float)infer_correct / batch * 100.0f << "%" << endl;

    AutogradContext::set_grad_enabled(true);

    // -------------------------------------------
    // CONV2D FORWARD TEST
    // -------------------------------------------
    cout << "\n--- Conv2D Test ---" << endl;

    int N = 1;
    int C_in = 1;
    int H = 4;
    int W = 4;

    float h_img[16] = {
         1, 2, 3, 4,
         5, 6, 7, 8,
         9,10,11,12,
        13,14,15,16
    };

    Tensor img(N * C_in * H * W, false);
    img.fromHost(h_img);

    Conv2D conv1(1, 2, 3, 3);
    conv1.set_input_dims(H, W);

    Tensor conv_out = conv1.forward(img, N);

    // Read back and print conv output
    float h_conv[8];
    conv_out.toHost(h_conv);

    int expected = N * 2 * 2 * 2;  // 1 * 2 channels * 2x2 output = 8
    printf("Conv2D output size = %d (expected %d)\n",
        conv_out.size, expected);
    printf("Conv2D output values:\n");
    printf("  Channel 0: %.4f %.4f %.4f %.4f\n",
        h_conv[0], h_conv[1], h_conv[2], h_conv[3]);
    printf("  Channel 1: %.4f %.4f %.4f %.4f\n",
        h_conv[4], h_conv[5], h_conv[6], h_conv[7]);

    return 0;
}







