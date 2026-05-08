# CUDA-DL-Mini-Deep-Learning-Framework
A lightweight deep learning framework built from scratch using CUDA C++, designed to demonstrate a deep understanding of GPU programming, neural network internals, and performance optimization.

🧠 Overview

This project implements the core components of modern deep learning frameworks like PyTorch and TensorFlow, but at a much lower level.
Instead of relying on high-level libraries, this framework directly uses CUDA kernels to perform tensor operations, enabling fine-grained control over computation and memory.

🎯 Key Features
⚡ Custom Tensor abstraction with GPU memory management
⚙️ CUDA-based implementations of:
Matrix multiplication (GEMM)
Activation functions (ReLU, Sigmoid)
Element-wise operations
🔁 Automatic differentiation engine (Autograd)
🧱 Modular neural network layers:
Fully Connected (Linear)
Activation layers
📉 Loss functions (MSE, Cross Entropy)
🔧 Optimizers (SGD, optional Adam)
🧪 End-to-end training pipeline on GPU

🏗️ Architecture
Tensor (GPU memory)
      ↓
CUDA Kernels (Compute)
      ↓
Autograd Engine (Gradient Flow)
      ↓
Layers (Linear, Activation)
      ↓
Model (Sequential API)
      ↓
Training Loop (Loss + Optimizer)

🔍 What This Project Demonstrates

💡 Deep Learning Fundamentals
Forward and backward propagation
Gradient computation using chain rule
Neural network training dynamics

⚡ GPU Programming Expertise
CUDA kernel design and execution
Thread/block hierarchy
Memory optimization (global vs shared memory)

🧠 Systems-Level Understanding
How frameworks like PyTorch actually work internally
Performance bottlenecks in deep learning workloads
Efficient tensor computation design

📊 Performance Focus
This project includes:

Benchmarking of naive vs optimized CUDA kernels
Comparison with cuBLAS/cuDNN (where applicable)
Profiling using NVIDIA Nsight tools

🚀 Applications
Educational tool for understanding deep learning internals
Foundation for building optimized inference engines
Useful for domains requiring low-latency GPU computation, such as:
Signal processing (IQ data, spectrograms)
Computer vision
Real-time AI systems

🛠️ Tech Stack
CUDA C++
NVIDIA CUDA Toolkit
(Optional) cuBLAS / cuDNN for benchmarking
Nsight Systems / Nsight Compute (profiling)


Structure:
include/ → declarations (headers)
src/ → implementations (.cu files)
main.cu → only for testing / training loop

<img width="1119" height="435" alt="image" src="https://github.com/user-attachments/assets/c29f342c-ea81-4fe9-b801-096e09950305" />
Fig 1: training loop output with MSE loss values included 

<img width="1103" height="621" alt="image" src="https://github.com/user-attachments/assets/39d5a365-d3ae-472a-aa78-34130f6e38df" />      <img width="1109" height="617" alt="image" src="https://github.com/user-attachments/assets/990fa919-ea00-49a7-8694-30ed7a9c0823" />
Fig 2: Training loop output for Input->Linear->ReLU->Linear layers with loss included: displays consistently decreasing loss and increasing output values

<img width="977" height="623" alt="image" src="https://github.com/user-attachments/assets/c84aab60-7649-46c8-8830-b3ca8a65d8b3" />
<img width="1099" height="620" alt="image" src="https://github.com/user-attachments/assets/f5ddb53a-c888-4484-9d52-9a6b8db85a14" />
Fig 3: Training loop output with softmax and cross entropy included, and random weight initialization included in linear layer -> faster error decrease and fster increasing output values.

MODEL ABSTRACTION:
Instead of writing layers manually like:
Tensor h1 = layer1.forward(x);
Tensor a1 = relu.forward(h1);
Tensor out = layer2.forward(a1);

we implement Sequential API Style like:
model.add(layer1);
model.add(relu);
model.add(layer2);
Tensor out = model.forward(x);

The framework then manages:
Layer ordering
Forward propagation
Backward propagation

Modular Deep Learning Architecture implemented with sequential container:
Sequential model;

model.add(&layer1);
model.add(&relu);
model.add(&layer2);
This creates the container and registers layers inside it.

The Sequential container acts as a Layer execution pipeline manager
It automates:

Forward propagation
Backward propagation
Layer ordering

Tensor Safety Upgrade:
When copying tensors, instead of shallow pointer copy like a=b, we implement:
Tensor& operator=(const Tensor& other) -> which allocates NEW GPU memory, copies GPU contents safely and avoids double free, also safe memory cleanup.








