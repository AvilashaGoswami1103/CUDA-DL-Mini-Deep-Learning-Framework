#include <iostream>
#include "tensor.h"
#include "linear_layer.h"

using namespace std;

int main() {
    int batch = 2;
    int in_f = 3;
    int out_f = 2;

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

    // 🔹 Forward pass
    Tensor out = layer.forward(x, batch);

    // 🔹 Allocate BEFORE using
    float* h_out = new float[batch * out_f];

    out.toHost(h_out);

    cout << "Output: ";
    for (int i = 0; i < batch * out_f; i++) {
        cout << h_out[i] << " ";
    }
    cout << endl;

    delete[] h_out;

    return 0;
}