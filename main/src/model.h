#pragma once

#include "pch.h"

// Unrolls a loop at compile-time
template <s64 First, s64 Last, typename Lambda>
void static_for(Lambda&& f)
{
    if constexpr (First < Last) {
        f(types::integral_constant<s64, First> {});
        static_for<First + 1, Last>(f);
    }
}

#include "activation.h"
#include "loss.h"

#include "dyn_mat.h"

using InputMat = matf<INPUT_SHAPE + 1, N_TRAIN_EXAMPLES_PER_STEP>;
using OutputMat = matf<OUTPUT_NEURONS, N_TRAIN_EXAMPLES_PER_STEP>;

struct base_layer {
    virtual void initialize_weights() {};

    virtual void feedforward(void* input, s64 input_dim) {};

    virtual void calculate_delta(void* next_layer_delta_weighted) {};

    virtual void update_weights(f32 learning_rate) {};

    virtual array<f32> get_packed_weights() { return {}; };

    struct information {
        s64 NumInputs;
        s64 NumNeurons;

        f32* Out;
        f32* DeltaWeighted;
    };

    // This just returns some important pointers to the cached calculations of a layer
    // and some info about the layer architeture since that is actually templated
    // and we lose that information when we cast to base_layer*.
    virtual information get_information() = 0;
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

struct dense_layer_neuron_info {
    s64 NumInputs;
    s64 NumNeurons;
};

template <dense_layer_neuron_info Info>
struct dense_layer : base_layer {
    static constexpr s64 NumInputs = Info.NumInputs;
    static constexpr s64 NumNeurons = Info.NumNeurons;

    matf<NumNeurons, NumInputs + 1> Weights; // +1 for the bias term
    activation_func_train<NumNeurons>* Activation;

    dense_layer(activation_func_train<NumNeurons>* activation)
        : Activation(activation)
    {
    }

    void initialize_weights() override
    {
        For_as(stripe, Weights.Stripes)
        {
            // 32767 is the max value of rand()
            For(stripe) it = ((f32)rand() / (32767 / 2)) - 1; // Generate a value between - 1 and 1... @Robustness, This is ugly and bad
        }
    }

    // The following are the cached variables that are used while training. These change from epoch to epoch.
    // When used later for prediction, these don't matter (only the weights and the activation function matter).

    matf<NumInputs + 1, N_TRAIN_EXAMPLES_PER_STEP>* In = null;

    // _Out_ is the output of each neuron for each training example.
    matf<NumNeurons + 1, N_TRAIN_EXAMPLES_PER_STEP> Out;

    // _Delta_ is calculated as:
    //    f'(Out) * cost_derivative              (when neurons are output for the model)
    //    f'(Out) * Delta_weighted,next_layer    (when neurons are hidden)
    //
    // Where f' is the derivative of the activation function.
    matf<NumNeurons, N_TRAIN_EXAMPLES_PER_STEP> Delta;

    // This is passed to the previous layer when doing backpropagation.
    // We do this in this layer because we already have the Delta and the Weights,
    // so we don't have to supply the weights to the previous layer.
    //
    // It is calculated by just dot(T(Weights), Delta).
    matf<NumInputs, N_TRAIN_EXAMPLES_PER_STEP> DeltaWeighted;

    void feedforward(void* input, s64 input_dim) override
    {
        assert(input_dim == NumInputs + 1); // Sanity

        auto* input_as_matrix = (matf<NumInputs + 1, N_TRAIN_EXAMPLES_PER_STEP>*)input;
        In = input_as_matrix; // Used later when calculating the weight update.

        Out.get_view<NumNeurons, N_TRAIN_EXAMPLES_PER_STEP>(1, 0) = dot(Weights, *In);
        Out.row(0) = vecf<N_TRAIN_EXAMPLES_PER_STEP>(1.0f); // Augment the output to account for the bias term

        Activation->apply_to_matrix(Out);
    }

    // We are going backwards, so _next_layer_delta_weighted_ is calculated before this layer's delta.
    void calculate_delta(void* next_layer_delta_weighted) override
    {
        // If this is the final layer _nextLayerDelta_ points to the derivative of the loss function,
        // otherwise it points to next layer's _DeltaWeighted_.
        auto* d = (matf<NumNeurons, N_TRAIN_EXAMPLES_PER_STEP>*)next_layer_delta_weighted;
        Delta = *d * Activation->get_derivative(Out);
        DeltaWeighted = dot(T(Weights), Delta).get_view<NumInputs, N_TRAIN_EXAMPLES_PER_STEP>(1, 0);
    }

    void update_weights(f32 learning_rate) override
    {
        auto average_kernel_update = dot(Delta, T(*In)) / (f32)N_TRAIN_EXAMPLES_PER_STEP;
        Weights -= learning_rate * average_kernel_update;
    }

    // Returns the trained weights but packed together (they might not be
    // because matrices may add padding to improve performance). Allocates an array.
    array<f32> get_packed_weights() override
    {
        matf<NumNeurons, NumInputs + 1, true> packed_weights = Weights;

        s64 num_floats = NumNeurons * (NumInputs + 1);

        array<f32> result;
        reserve(result, num_floats);

        copy_memory(result.Data, &packed_weights.Stripes[0][0], num_floats * sizeof(f32));
        result.Count = num_floats;

        return result;
    }

    information get_information() override
    {
        information result;
        result.NumInputs = NumInputs;
        result.NumNeurons = NumNeurons;

        result.Out = &Out.Stripes[0][0];
        result.DeltaWeighted = &DeltaWeighted.Stripes[0][0];

        result.Activation = Activation;

        return result;
    }
};

struct model {
    InputMat* Input;
    OutputMat* Targets;

    array<base_layer*> Layers;

    array<dyn_mat> PackedWeights;
    bool DirtyPackedWeights = true;

    loss_function Loss;

    f32 LearningRate;
    s64 Epochs;
};

// This is used for an argument to build_training_context for a nice syntactic
// sugar that makes it obvious which parameter we are setting.
struct hyper_parameters {
    array<base_layer*> Layers;
    f32 LearningRate;
    loss_function Loss;
};

inline model compile_model(hyper_parameters h_params)
{
    model result;

    // @Volatile: As we add more types of layers we check the compatibility here.
    // For now, only dense layers exist.
    assert(h_params.Layers.Count >= 1);

    assert(h_params.Loss.Calculate != null && "Forgot loss function");

    result.Layers = h_params.Layers;
    result.Loss = h_params.Loss;
    result.LearningRate = h_params.LearningRate;

    For(result.Layers) it->initialize_weights();

    return result;
}

inline void train(model m, s64 epochs)
{
    m.DirtyPackedWeights = true;

    For_as(epoch, range(epochs))
    {
        void* input = m.Input;
        s64 input_dim = INPUT_SHAPE + 1; // +1 for the bias term

        For_enumerate_as(layer_index, layer, m.Layers)
        {
            // The last layer doesn't do activation.
            layer->feedforward(input, input_dim);

            auto info = layer->get_information(); // Cache this?
            input = info.Out;
            input_dim = info.NumNeurons + 1; // +1 for the bias term
        }

        auto out_layer = m.Layers[-1];
        auto out_info = out_layer->get_information();

        assert(out_info.NumNeurons == OUTPUT_NEURONS); // Sanity

        auto* out_as_matrix = (matf<OUTPUT_NEURONS + 1, N_TRAIN_EXAMPLES_PER_STEP>*)out_info.Out;
        // auto predicted = out_as_matrix->get_view<OUTPUT_NEURONS, N_TRAIN_EXAMPLES_PER_STEP>(1, 0);

        OutputMat losses;
        For_enumerate(losses.Stripes) it = m.Loss.Calculate(m.Targets->Stripes[it_index], out_as_matrix->Stripes[it_index + 1]);

        auto costMatrix = T(losses);

        auto cost = costMatrix.Stripes[0];
        For(range(1, costMatrix.R)) cost += costMatrix.Stripes[it];
        cost /= (f32)N_TRAIN_EXAMPLES_PER_STEP;

        if (epoch == 700) {
            int a = 42;
        }

        fmt::print("Cost: {}\n", cost);

        // cost derivative
        OutputMat d;
        For_enumerate(d.Stripes) it = m.Loss.GetDerivative(m.Targets->Stripes[it_index], out_as_matrix->Stripes[it_index + 1]);

        void* delta = &d;

        // Calculate delta for all layers going backwards
        For(range(m.Layers.Count - 1, -1, -1))
        {
            auto l = m.Layers[it];

            l->calculate_delta(delta);
            delta = l->get_information().DeltaWeighted;
        }

        // Finally, do the weight updates
        For(m.Layers) it->update_weights(m.LearningRate);
    }
}

// We do this for syntactic sugar..
struct fit_model_parameters {
    array_view<f32> X;
    array_view<f32> y;

    s64 Epochs;
};

inline void fit_model(model m, fit_model_parameters params)
{
    assert(params.X.Count % N_TRAIN_EXAMPLES_PER_STEP == 0 && "Bad X shape");
    assert(params.X.Count / N_TRAIN_EXAMPLES_PER_STEP == INPUT_SHAPE && "Bad X shape");

    assert(params.y.Count % N_TRAIN_EXAMPLES_PER_STEP == 0 && "Bad y shape");
    assert(params.y.Count / N_TRAIN_EXAMPLES_PER_STEP == OUTPUT_NEURONS && "Bad y shape");

    // We copy the inputs and the targets so the memory
    // is guaranteed to be next to each other and properly packed!

    // We need to augment the inputs to include a "1" (for the bias term).
    // And we need to transpose the matrix before feeding in the model.
    matf<N_TRAIN_EXAMPLES_PER_STEP, INPUT_SHAPE + 1> Xt;

    // We need to copy row by row because of the packing
    auto* p = params.X.Data;
    For(Xt.Stripes)
    {
        it.Data[0] = 1.0f;
        copy_memory(&it.Data[1], p, INPUT_SHAPE * sizeof(f32));
        p += INPUT_SHAPE;
    }

    // Transpose it
    auto* X = new InputMat;
    *X = T(Xt);

    auto* y = new OutputMat;
    p = params.y.Data;
    For(y->Stripes)
    {
        copy_memory(&it.Data[0], p, y->C * sizeof(f32));
        p += y->C;
    }

    m.Input = X;
    m.Targets = y;

    train(m, params.Epochs);
}

inline array<vecf<OUTPUT_NEURONS>> predict(model m, array_view<f32> X)
{
    assert(X.Count % INPUT_SHAPE == 0 && "Bad X shape");

    if (m.DirtyPackedWeights) {
        For(m.PackedWeights) free(it.Data);
        free(m.PackedWeights);

        For(m.Layers)
        {
            auto packed_weights = it->get_packed_weights();
            auto info = it->get_information();
            append(m.PackedWeights, dyn_mat { info.NumNeurons, info.NumInputs + 1, packed_weights });
        }
    }

    dyn_mat Xdyn;
    append_array(Xdyn.Data, X);
    Xdyn.R = X.Count / INPUT_SHAPE;
    Xdyn.C = INPUT_SHAPE;

    dyn_mat Xt = augment_dyn_mat(Xdyn);

    auto Xaugmented = T(Xt);
    free(Xt.Data);

    auto input = Xaugmented;

    For_enumerate_as(layer_index, layer, m.Layers)
    {
        auto weights = m.PackedWeights[layer_index];

        auto out = dot(weights, input);

        auto info = layer->get_information();
        For(out.Data) it = info.Activation->apply_single(it);

        if (layer_index != m.Layers.Count - 1) {
            auto tout = T(out);
            auto augmented_tout = augment_dyn_mat(tout);

            input = T(augmented_tout);

            free(out.Data);
            free(tout.Data);
            free(augmented_tout.Data);
        } else {
            input = out;
        }
    }

    auto output = input;

    auto tout = T(output);
    free(output.Data);

    array<vecf<OUTPUT_NEURONS>> y;
    reserve(y, X.Count);

    For(range(tout.R))
    {
        vecf<OUTPUT_NEURONS> o;
        copy_memory(&o, &tout.Data[it * tout.C], OUTPUT_NEURONS * sizeof(f32));
        append(y, o);
    }
    return y;
}

template <s64... Neurons>
struct dense_layer_architecture {
    stack_array<s64, sizeof...(Neurons)> StackNeurons = { Neurons... };

    constexpr dense_layer_architecture() { }

    constexpr dense_layer_neuron_info operator[](s64 index) const
    {
        assert(index >= 0);
        assert(index < StackNeurons.Count);

        s64 prev = StackNeurons[index];
        s64 curr = StackNeurons[index + 1];
        return { prev, curr };
    }
};