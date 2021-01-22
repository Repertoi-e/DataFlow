#pragma once

#include "../pch.h"

using BatchInputMat = matf<INPUT_SHAPE + 1, N_TRAIN_EXAMPLES_PER_STEP>;
using BatchOutputMat = matf<OUTPUT_NEURONS, N_TRAIN_EXAMPLES_PER_STEP>;

struct loss_function {
    using func_train_t = BatchOutputMat (*)(const BatchOutputMat & y, const BatchOutputMat& z);
    using func_derivative_t = BatchOutputMat (*)(const BatchOutputMat& y, const BatchOutputMat& z);

    // This is a function pointer which calculates the loss
    // function and returns the result (for each training sample, in a vector).
    func_train_t Calculate = null;

    // This is a function pointer which calculates the derivative of
    // the loss function and returns the result (for each training sample, in a vector).
    func_derivative_t GetDerivative = null;
};

// _y_ is the target.
// _z_ is an output of a set of neurons (should be between 0 and 1).
inline auto binary_cross_entropy(const BatchOutputMat& y, const BatchOutputMat& z)
{
    // We avoid calculating log(0) by adding a small epsilon.
    // This means that the loss will be slighly higher than it can be in reality.
    return -y * log(1e-07F + z) - (1.0f - y) * log(1.0f - z + 1e-07F); // @Performance @Math
}

// _y_ is the target.
// _z_ is an output of a set of neurons (should be between 0 and 1).
inline auto binary_cross_entropy_derivative(const BatchOutputMat& y, const BatchOutputMat& z)
{
    // Avoid division by zero by adding a small epsilon
    return -y / (z + 1e-07F) + (1.0f - y) / (1.0f - z + 1e-07F);
}

// Used when there is a single output neuron and two classes (0 and 1).
constexpr loss_function BinaryCrossEntropy = { binary_cross_entropy, binary_cross_entropy_derivative };
