#pragma once
#include "tensor.h"
#include <vector>
#include <string>

// Saves all parameter data to a binary file
void save_checkpoint(const std::vector<Tensor*>& params,
    const std::string& path);

// Loads parameter data from a binary file into existing tensors
// Tensors must already be allocated with the correct sizes
void load_checkpoint(const std::vector<Tensor*>& params,
    const std::string& path);
