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

<img width="1112" height="420" alt="image" src="https://github.com/user-attachments/assets/49fbe8e0-01f4-4f72-8a6a-825369d4a0e6" />
Fig 2: training loop output with linear -> ReLU layers


