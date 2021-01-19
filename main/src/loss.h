#pragma once

#include "pch.h"

// This is to silence annoying errors, these don't fire if you define them before including this file.
#ifndef INPUT_SHAPE
#define INPUT_SHAPE 1
#endif

#ifndef N_TRAIN_EXAMPLES_PER_STEP
#define N_TRAIN_EXAMPLES_PER_STEP 1
#endif

#ifndef OUTPUT_NEURONS
#define OUTPUT_NEURONS 1
#endif

using LossesMat = matf<OUTPUT_NEURONS, N_TRAIN_EXAMPLES_PER_STEP>;

struct loss_function {
    using train_vec_t = vecf<N_TRAIN_EXAMPLES_PER_STEP>;

    using func_train_t = train_vec_t (*)(const train_vec_t& y, const train_vec_t& z);
    using func_derivative_t = train_vec_t (*)(const train_vec_t& y, const train_vec_t& z);

    // This is a function pointer which calculates the loss
    // function and returns the result (for each training sample, in a vector).
    func_train_t Calculate = null;

    // This is a function pointer which returns the derivative of
    // the loss function and returns the result (for each training sample, in a vector).
    func_derivative_t GetDerivative = null;
};

//
// This is also known as binary cross-entropy.
//
// _y_ is the target.
// _z_ is an output of a set of neurons (should be from 0 to 1).
//
// We avoid calculating log(0) by adding a small epsilon. 
// This means that the loss will be slighly higher than it can be in reality.
inline auto binary_cross_entropy(const loss_function::train_vec_t& y, const loss_function::train_vec_t& z)
{
    // @Performance @Math
    return -y * element_wise_log(1e-07F + z) - (1.0f - y) * element_wise_log(1.0f - z + 1e-07F);
}

inline auto binary_cross_entropy_derivative(const loss_function::train_vec_t& y, const loss_function::train_vec_t& z)
{
    // Avoid division by zero by adding a small epsilon
    return -y / (z + 1e-07F) + (1.0f - y) / (1.0f - z + 1e-07F);
}

// Used when there is a single output neuron and two classes (0 and 1).
constexpr loss_function BinaryCrossEntropy = { binary_cross_entropy, binary_cross_entropy_derivative };
