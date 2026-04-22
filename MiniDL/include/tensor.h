#pragma once    
//Ensures this header file is included only once per compilation unit, prevents multiple definition errors
#include <cuda_runtime.h>

class Tensor {
public: //everything is accessible from outside the class
    float* data;    //pointer to GPU memory holding actual tensor values
    float* grad;    //pointer to GPU memory holding gradients
    int size;
    bool requires_grad; //Indicates whether gradients should be computed

    Tensor(int size, bool requires_grad = false);
    //constructor declaration

    void fromHost(float* h_data);   //CPU->GPU
    void toHost(float* h_data);     //gpu->cpu
    void zero_grad();   // Sets gradient values to zero(befoe backpropagation)

    ~Tensor();  //destrcutor declaration
    Tensor(const Tensor& other);  // Copy constructor
};
