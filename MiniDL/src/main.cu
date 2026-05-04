#include <iostream>
#include "tensor.h"
#include "linear_layer.h"
#include "optimizer.h"
#include "loss.h"
#include "relu.h"
#include "softmax.h"
#include "cross_entropy.h"
#include "sequential.h"

using namespace std;

int main() {
    int batch = 2;  //batch size = 2 samples
    int in_f = 3;   //input features = 3 per sample 
    int out_f = 2;  //output features = 2 per samples

    float h_input[] = {
        1, 2, 3,
        4, 5, 6
    };

    Tensor x(batch * in_f, false);
    x.fromHost(h_input);

    //Linear layer(in_f, out_f);
    // implementing multi-layer (MLP)
    int hidden_f = 4;   // hidden shape = 4 ( 4 per sample)

    Linear layer1(in_f, hidden_f);
    ReLU relu;
    Linear layer2(hidden_f, out_f);

    Sequential model;

    model.add(&layer1);
    model.add(&relu);
    model.add(&layer2);

    Softmax softmax (out_f);
    CrossEntropyLoss loss_fn;

    // 🔹 Debug input
    float temp[6];
    x.toHost(temp);

    cout << "Input: ";
    for (int i = 0; i < 6; i++) cout << temp[i] << " ";
    cout << endl;

    // 🔥 ADD TRAINING LOOP HERE
    int epochs = 20;
    SGD optimizer(0.01f); // Creates an SGD optimizer with learning rate 0.01.

    for (int epoch = 0; epoch < epochs; epoch++) {

        layer1.W->zero_grad();
        layer1.b->zero_grad();
        layer2.W->zero_grad();
        layer2.b->zero_grad();

        // Forward
        Tensor logits = model.forward(x, batch);

        Tensor out = softmax.forward(
            logits,
            batch
        );
        // Input → hidden representation → output

        float* h_out = new float[batch * out_f];
        out.toHost(h_out);
        //copy back to CPU

        // Target
        // using one-hot targets
        float h_target[] = {
                1, 0,
                0, 1
        };
        // Meaning: sample 1 -> class 0
        // Meaning: sample 2 -> class 1

        // copy to GPU tensor target
        Tensor target(batch * out_f, false);
        target.fromHost(h_target);

        // MSE Loss
        float loss = loss_fn.forward(
            out,
            target,
            batch,
            out_f
        );

        Tensor d_out = loss_fn.backward(out, target);

        // Backward
        Tensor dX = model.backward(d_out, batch);

        // Update
        optimizer.step(layer1.W);
        optimizer.step(layer1.b);

        optimizer.step(layer2.W);
        optimizer.step(layer2.b);
        // Calls the SGD optimizer to update the weights and biases
      

        cout << "Epoch " << epoch << " Loss: " << loss << " Output: ";
        for (int i = 0; i < batch * out_f; i++)
            cout << h_out[i] << " ";
        cout << endl;

        delete[] h_out;
    }

    return 0;
}