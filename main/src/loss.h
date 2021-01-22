#pragma once

#include "pch.h"

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
// _z_ is an output of a set of neurons (should be from 0 to 1).
//
// We avoid calculating log(0) by adding a small epsilon. 
// This means that the loss will be slighly higher than it can be in reality.
inline auto binary_cross_entropy(const BatchOutputMat& y, const BatchOutputMat& z)
{
    auto eps_mat = BatchOutputMat(1e-07F);
    auto ones_mat = BatchOutputMat(1.0f);

    // @Performance @Math
    return -y * element_wise_log(eps_mat + z) - (ones_mat - y) * element_wise_log(ones_mat - z + eps_mat);
}

inline auto binary_cross_entropy_derivative(const BatchOutputMat& y, const BatchOutputMat& z)
{
    auto eps_mat = BatchOutputMat(1e-07F);
    auto ones_mat = BatchOutputMat(1.0f);

    // Avoid division by zero by adding a small epsilon
    return -y / (z + eps_mat) + (ones_mat - y) / (ones_mat - z + eps_mat);
}

// Used when there is a single output neuron and two classes (0 and 1).
constexpr loss_function BinaryCrossEntropy = { binary_cross_entropy, binary_cross_entropy_derivative };
