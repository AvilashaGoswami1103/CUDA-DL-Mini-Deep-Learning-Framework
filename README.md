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




