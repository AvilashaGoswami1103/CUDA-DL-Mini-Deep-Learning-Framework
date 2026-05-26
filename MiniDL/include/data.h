#pragma once

// 32 samples, 3 features each
// Classes: 0 if sum of features < 9, class 1 if sum >= 9
// This creates a linearly separable binary classification problem

static float train_data[32 * 3] = {
    // Class 0 (sum < 9)
    1.0f, 1.0f, 1.0f,
    1.0f, 2.0f, 1.0f,
    2.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 2.0f,
    2.0f, 2.0f, 1.0f,
    1.0f, 2.0f, 2.0f,
    2.0f, 1.0f, 2.0f,
    2.0f, 2.0f, 2.0f,
    1.0f, 1.0f, 3.0f,
    1.0f, 3.0f, 1.0f,
    3.0f, 1.0f, 1.0f,
    1.0f, 2.0f, 3.0f,
    2.0f, 3.0f, 1.0f,
    3.0f, 1.0f, 2.0f,
    1.0f, 3.0f, 3.0f,
    3.0f, 3.0f, 1.0f,
    // Class 1 (sum >= 9)
    3.0f, 3.0f, 3.0f,
    4.0f, 3.0f, 3.0f,
    3.0f, 4.0f, 3.0f,
    3.0f, 3.0f, 4.0f,
    4.0f, 4.0f, 3.0f,
    3.0f, 4.0f, 4.0f,
    4.0f, 3.0f, 4.0f,
    4.0f, 4.0f, 4.0f,
    5.0f, 2.0f, 3.0f,
    2.0f, 5.0f, 3.0f,
    3.0f, 2.0f, 5.0f,
    5.0f, 3.0f, 2.0f,
    4.0f, 5.0f, 2.0f,
    2.0f, 4.0f, 5.0f,
    5.0f, 4.0f, 3.0f,
    3.0f, 5.0f, 4.0f
};

// One-hot encoded labels
static float train_labels[32 * 2] = {
    // Class 0 → [1, 0]
    1,0,  1,0,  1,0,  1,0,
    1,0,  1,0,  1,0,  1,0,
    1,0,  1,0,  1,0,  1,0,
    1,0,  1,0,  1,0,  1,0,
    // Class 1 → [0, 1]
    0,1,  0,1,  0,1,  0,1,
    0,1,  0,1,  0,1,  0,1,
    0,1,  0,1,  0,1,  0,1,
    0,1,  0,1,  0,1,  0,1
};
