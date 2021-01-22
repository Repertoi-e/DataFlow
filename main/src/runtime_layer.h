#pragma once

#include "activation.h"

struct base_layer_runtime {
};

//
// NOTE!
//
// We require the model architecture to be known at compile-time so we can (and
// the compiler) do optimizations which otherwise wouldn't be possible. Since we
// plan on calling C++ from Python, one can be sophisticated and actually
// write a Python script which places a runtime parameter in the source code
// and compiles the program, and then calls it from Python.
//
// That way we get the best of both worlds. Python is king
// when it comes to automatization, C++ is king in performance.
//
//
// For the math we use the following notation:
//
// Input: X        with shape i
// Targets: y      with shape k
// Predicted: y~
//
// Layer 1 has j neurons - a_j.
//
// Weights from input to layer 1: W_ji  (for jth neuron, with ith input)
//
// Layer 2 has k neurons - a_k.
// .. and so on.
//
// So the flow for a two layer architecture is this:
// X -> Z_j = W_ji * X + b_j
//   -> A_j = f1(Z_j)
//   -> Z_k = W_kj * A_j + b_k
//   -> A_k = f2(Z_k)
//   -> y~ = A_k
//
// Where f1 and f2 are activation functions.
//
// Loss is calculated using y and y~.
//
// For backpropagation:
//   We calculate the cost derivative and pass that to calculate_delta for the output layer.
//   Each layer calculates its own Delta and DeltaWeigthed (where DeltaWeighted is just Delta
//   multiplied by the Weights between that layer and the previous).
//   DeltaWeighted is then passed to the previous layer and so on.
//
//   The change in weights is just: dot(Delta, T(In)) / N_TRAIN_EXAMPLES_PER_STEP,
//   where In is the input to this layer which is cached (we actually don't copy it but store
//   it just as a pointer to _Out_ of the previous layer, since we know that while backpropagating
//   we wouldn't mess with the outputs of neurons).
//
// Beautiful illustration: https://miro.medium.com/max/2400/1*dx2AYvXVyPZ38TAiPeD9Aw.jpeg <3
//
template <s64 NumInputs, s64 NumNeurons, typename _Activation>
struct dense_layer_runtime : base_layer_runtime {
    static constexpr s64 NUM_INPUTS = NumInputs;
    static constexpr s64 NUM_NEURONS = NumNeurons;

    using Activation = _Activation;

    matf<NUM_NEURONS, NUM_INPUTS + 1> Weights; // +1 for the bias term

    // The following are the cached variables that are used while training. These change from epoch to epoch.
    // When used later for prediction, these don't matter (only the weights matter).

    // _Out_ is the output of each neuron for each training example.
    matf<NUM_NEURONS + 1, N_TRAIN_EXAMPLES_PER_STEP> Out;

    // _Delta_ is calculated as:
    //    f'(Out) * cost_derivative              (when neurons are output for the model)
    //    f'(Out) * Delta_weighted,next_layer    (when neurons are hidden)
    //
    // Where f' is the derivative of the activation function.
    matf<NUM_NEURONS, N_TRAIN_EXAMPLES_PER_STEP> Delta;

    // This is passed to the previous layer when doing backpropagation.
    // We do this in this layer because we already have the Delta and the Weights,
    // so we don't have to supply the weights to the previous layer.
    //
    // It is calculated by just dot(T(Weights), Delta).
    matf<NUM_INPUTS, N_TRAIN_EXAMPLES_PER_STEP> DeltaWeighted;
};

template <s64 I>
using _ATE = std::tuple_element_t<I, ARCHITECTURE_T>;

template <s64 I>
struct dense_layer_runtime_lookup {
    using type = typename dense_layer_runtime<_ATE<I - 1>::NUM_NEURONS, _ATE<I>::NUM_NEURONS, typename _ATE<I>::Activation>;
};

// We specialize for the layer at index "0" (the input layer). TECHNICALLY we can't interpret that layer as a dense_layer since
// it doesn't play any role in the architecture other than to specify the input shape. Nevertheless we can treat it as a dense
// layer with no inputs and no activation in order to simplify our code later. Once we add more types of layers we should
// really stop refering to this as a "dense layer lookup".
template <>
struct dense_layer_runtime_lookup<0> {
    using type = dense_layer_runtime<1, _ATE<0>::NUM_NEURONS, decltype(void())>;
};

// We specialize for the layer at index "ARCHITECTURE_COUNT" (the POST output layer). TECHNICALLY we can't interpret that layer
// as a dense_layer since because it makes no sense. Nevertheless we can treat it as a dense layer which has a _DeltaWeighted_
// which has the same type as the derivative of our cost function. This simplifies our code in the training loop. Once we add
// more types of layers we should really stop refering to this as a "dense layer lookup".
template <>
struct dense_layer_runtime_lookup<ARCHITECTURE_COUNT> {
    using type = dense_layer_runtime<_ATE<ARCHITECTURE_COUNT - 1>::NUM_NEURONS, 1, decltype(void())>;
};

// Short-hand. Returns the proper templated type of dense_layer_runtime without having to do ARCHITECTURE each time.
template <s64 I>
using DLR_T = typename dense_layer_runtime_lookup<I>::type;

// Here we also check for compatibility between layers..
auto get_layers_from_architecture()
{
    static_assert(ARCHITECTURE_COUNT > 1, "At least one input and one other layer is required");

    array<base_layer_runtime*> layers;
    static_for<0, ARCHITECTURE_COUNT>([&](auto it) {
        using T = _ATE<it>;

        if constexpr (it == 0) {
            static_assert(T::TYPE == layer_type::INPUT, "First layer needs to be input");
            append(layers, null); // We append a null pointer for the input layer because it doesn't do anything but fuck-up our indexing later.
        }
        if constexpr (it > 0) {
            static_assert(T::TYPE != layer_type::INPUT, "Only one input layer is allowed");
            append(layers, new DLR_T<it>);
        }
    });
    return layers;
}

