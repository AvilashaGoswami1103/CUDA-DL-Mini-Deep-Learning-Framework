#include <iostream>
#include "tensor.h"
#include "linear_layer.h"
#include "optimizer.h"

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

    Linear layer(in_f, out_f);

    // 🔹 Debug input
    float temp[6];
    x.toHost(temp);

    cout << "Input: ";
    for (int i = 0; i < 6; i++) cout << temp[i] << " ";
    cout << endl;

   // // 🔹 Forward pass
   // Tensor out = layer.forward(x, batch);

   // // 🔹 Allocate BEFORE using
   // float* h_out = new float[batch * out_f];

   // out.toHost(h_out);

   // cout << "Output: ";
   // for (int i = 0; i < batch * out_f; i++) {
   //     cout << h_out[i] << " ";
   // }
   // cout << endl;

   // delete[] h_out;

   // //Backpropagation test

   // // Fake gradient (like loss gradient)
   // float h_dout[] = {
   //     1, 1,
   //     1, 1
   // };
   // // this creates a small array on the host(CPU) called h_dout
   // // simulating (∂L/∂Y) that would normally come from the loss function, just filled here for simplicity
   // // Shape-wise: batch=2, out_f=2, 2x2 matrix flattened

   // Tensor d_out(batch * out_f, false);
   // // Allocate a Tensor object d_out on the device (GPU) with size batch × out_features
   //
   // d_out.fromHost(h_dout);

   // // Run backward
   // Tensor dX = layer.backward(d_out, batch);
   // // Call the backward method of the layer (your Linear layer), runs the CUDA kernels for dW, db and dX.
   //// The function returns dX (∂L/∂X), the gradient wrt the input.

   // // Print dX
   // float* h_dX = new float[batch * in_f];
   // // Allocate a CPU array h_dX to hold the values of dX
   // dX.toHost(h_dX);

   // cout << "dX: ";
   // for (int i = 0; i < batch * in_f; i++) {
   //     cout << h_dX[i] << " ";
   // }
   // cout << endl;

   // delete[] h_dX;

   // return 0;

    // 🔥 ADD TRAINING LOOP HERE
    int epochs = 5;
    SGD optimizer(0.01f); // Creates an SGD optimizer with learning rate 0.01.

    for (int epoch = 0; epoch < epochs; epoch++) {

        // Forward
        Tensor out = layer.forward(x, batch);
        //runs forward pass: out=XW+b

        float* h_out = new float[batch * out_f];
        out.toHost(h_out);
        //copy back to CPU

        // Fake gradient
        float h_dout[] = {
            1, 1,
            1, 1
        };

        // copy to GPU tensor d_out
        Tensor d_out(batch * out_f, false);
        d_out.fromHost(h_dout);

        // Backward
        Tensor dX = layer.backward(d_out, batch);
        //runs backward pass, Computes gradients wrt weights (dW), bias (db), and input (dX)

        // Update
        optimizer.step(layer.W);
        optimizer.step(layer.b);
        // Calls the SGD optimizer to update the weights and biases
      

        cout << "Epoch " << epoch << " Output: ";
        for (int i = 0; i < batch * out_f; i++)
            cout << h_out[i] << " ";
        cout << endl;

        delete[] h_out;
    }

    return 0;
}