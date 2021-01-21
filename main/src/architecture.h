#pragma once

#include "activation.h"

enum class layer_type {
    INPUT,
    DENSE
};

template <s64 NumInputs>
struct input {
    static constexpr auto TYPE = layer_type::INPUT;
    static constexpr s64 NUM_NEURONS = NumInputs;
};

template <s64 NumNeurons, typename _Activation = activation_none>
struct dense {
    static constexpr auto TYPE = layer_type::DENSE;

    static constexpr s64 NUM_NEURONS = NumNeurons;

    using Activation = _Activation;
};
