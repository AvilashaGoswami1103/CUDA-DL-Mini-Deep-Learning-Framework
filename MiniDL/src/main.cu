#include <iostream>
#include "tensor.h"
#include "linear_layer.h"

using namespace std;

int main() {
    int batch = 2;
    int in_f = 3;   //input shape: 2 x 3
    int out_f = 2;  //output shape: 2 x 2

    float h_input[] = {
        1, 2, 3,
        4, 5, 6
    };

    Tensor x(batch * in_f, false);  // creates a tensor named x : 2x3
    x.fromHost(h_input);

    // creates Linear Layer
    Linear layer(in_f, out_f);

    // Forward passes the linear layer
    Tensor out = layer.forward(x, batch);
    // out = xW + b

    float* h_out = new float[batch * out_f]; //space on CPU [2x2]
    out.toHost(h_out);

    for (int i = 0; i < batch * out_f; i++) {
        cout << h_out[i] << " ";
    }

    cout << endl;

    delete[] h_out;

    return 0;
}