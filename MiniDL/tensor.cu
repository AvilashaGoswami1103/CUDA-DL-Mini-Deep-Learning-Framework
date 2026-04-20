//Basic Tensor Code

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

class Tensor {
public:
    float* d_data;   // Pointer to GPU memory, lives in GPU memory
    int size;        // Total number of elements

    // Constructor
    Tensor(int size) {
        this->size = size;  
        // this->size = size; → Inside the constructor, 
        // the parameter size (the one passed in when you create the object) is assigned to the class’s member variable size

        // Allocate memory on GPU
        cudaMalloc(&d_data, size * sizeof(float));
    }

    // Copy data from CPU to GPU
    void fromHost(float* h_data) {
        cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Copy data from GPU to CPU
    void toHost(float* h_data) {
        cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Destructor (free GPU memory)
    ~Tensor() {
        cudaFree(d_data);
    }
};

// Utility function to print tensor
void printTensor(Tensor& t) {
    float* h_data = new float[t.size];

    t.toHost(h_data);

    for (int i = 0; i < t.size; i++) {
        cout << h_data[i] << " ";
    }
    cout << endl;

    delete[] h_data;
}

// Test main
int main() {
    int N = 10;

    // Create CPU data
    float h_data[10];
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 1.0f;
    }

    // Create Tensor
    Tensor t(N);

    // Copy to GPU
    t.fromHost(h_data);

    // Print (copy back from GPU)
    printTensor(t);

    return 0;
}