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

using BatchInputMat = matf<INPUT_SHAPE + 1, N_TRAIN_EXAMPLES_PER_STEP>;
using BatchOutputMat = matf<OUTPUT_NEURONS, N_TRAIN_EXAMPLES_PER_STEP>;

struct base_layer_runtime {
    virtual void initialize_weights() {};

    virtual void feedforward(void* input) {};

    virtual void calculate_delta(void* next_layer_delta_weighted) {};

    virtual void update_weights(f32 learning_rate) {};
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
template <s64 NumInputs, s64 NumNeurons, typename Activation>
struct dense_layer_runtime : base_layer_runtime {
    matf<NumNeurons, NumInputs + 1> Weights; // +1 for the bias term

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

    void feedforward(void* input) override
    {
        auto* input_as_matrix = (matf<NumInputs + 1, N_TRAIN_EXAMPLES_PER_STEP>*)input;
        In = input_as_matrix; // Used later when calculating the weight update.

        Out.get_view<NumNeurons, N_TRAIN_EXAMPLES_PER_STEP>(1, 0) = dot(Weights, *In);
        Out.row(0) = vecf<N_TRAIN_EXAMPLES_PER_STEP>(1.0f); // Augment the output to account for the bias term

        Activation::apply(Out);
    }

    // We are going backwards, so _next_layer_delta_weighted_ is calculated before this layer's delta.
    void calculate_delta(void* next_layer_delta_weighted) override
    {
        // If this is the final layer _nextLayerDelta_ points to the derivative of the loss function,
        // otherwise it points to next layer's _DeltaWeighted_.
        auto* d = (matf<NumNeurons, N_TRAIN_EXAMPLES_PER_STEP>*)next_layer_delta_weighted;

        decltype(Delta) outd = Out.get_view<NumNeurons, N_TRAIN_EXAMPLES_PER_STEP>(1, 0);

        Delta = *d * Activation::get_derivative(outd);
        DeltaWeighted = dot(T(Weights), Delta).get_view<NumInputs, N_TRAIN_EXAMPLES_PER_STEP>(1, 0);
    }

    void update_weights(f32 learning_rate) override
    {
        auto average_kernel_update = dot(Delta, T(*In)) / (f32)N_TRAIN_EXAMPLES_PER_STEP;
        Weights -= learning_rate * average_kernel_update;
    }
};

struct model {
    array<base_layer_runtime*> Layers;

    f32 LearningRate;

    loss_function Loss;

    array<std::tuple<BatchInputMat*, BatchOutputMat*>> Batches;
};

// Here we also check for compatibility between layers
auto get_layers_from_architecture()
{
    static_assert(ARCHITECTURE_COUNT > 1, "At least one input and one other layer is required");

    array<base_layer_runtime*> layers;
    static_for<0, ARCHITECTURE_COUNT>([&](auto it) {
        using T = std::tuple_element_t<it, ARCHITECTURE_T>;

        if constexpr (it == 0) {
            static_assert(T::TYPE == layer_type::INPUT, "First layer needs to be input");
        }
        if constexpr (it > 0) {
            static_assert(T::TYPE != layer_type::INPUT, "Only one input layer is allowed");

            using TM1 = std::tuple_element_t<it - 1, ARCHITECTURE_T>;

            append(layers, new dense_layer_runtime<TM1::NUM_NEURONS, T::NUM_NEURONS, T::Activation>);
        }
    });
    return layers;
}

// This is used for an argument to _compile_model_ for a nice syntactic
// sugar that makes it obvious which parameter we are setting.
struct hyper_parameters {
    f32 LearningRate;
    loss_function Loss;
};

// This returns a new model each time.
inline model compile_model(hyper_parameters h_params)
{
    model result;

    // Layer initialization
    auto layers = get_layers_from_architecture();
    For(layers) it->initialize_weights();
    result.Layers = layers;

    // Rest of hyper-parameters
    assert(h_params.Loss.Calculate != null && "Forgot loss function");
    result.Loss = h_params.Loss;
    result.LearningRate = h_params.LearningRate;

    return result;
}

inline void train(model m, s64 epochs)
{
    For_as(epoch, range(epochs))
    {
        fmt::print("Epoch {}/{}\n", epoch + 1, epochs);

        f32 eq_per_batch = 40.0f / m.Batches.Count;
        For_enumerate_as(batch_index, batch, m.Batches)
        {
            fmt::print("{}/{} ", batch_index * N_TRAIN_EXAMPLES_PER_STEP, m.Batches.Count * N_TRAIN_EXAMPLES_PER_STEP);
            fmt::print("[{:=<{}}>{:.<{}}]\r", "", (s64) (batch_index * eq_per_batch), "", (s64) max(eq_per_batch * m.Batches.Count - batch_index * eq_per_batch - 1, 0.0f));

            auto [attributes, targets] = batch;

            void* o = attributes; // Holds the last output of the layers

            static_for<1, ARCHITECTURE_COUNT>([&](auto i) {
                using T = std::tuple_element_t<i, ARCHITECTURE_T>;
                using TM1 = std::tuple_element_t<i - 1, ARCHITECTURE_T>;

                using L_T = dense_layer_runtime<TM1::NUM_NEURONS, T::NUM_NEURONS, T::Activation>;
                auto l = (L_T*)m.Layers[i - 1]; // -1 for the first layer (which is the input layer)
                l->feedforward(o);
                o = &l->Out;
            });

            // Get the last layer
            using _T = std::tuple_element_t<ARCHITECTURE_COUNT - 1, ARCHITECTURE_T>;
            using _TM1 = std::tuple_element_t<ARCHITECTURE_COUNT - 2, ARCHITECTURE_T>;

            using O_T = dense_layer_runtime<_TM1::NUM_NEURONS, _T::NUM_NEURONS, _T::Activation>;
            auto* ol = (O_T*)m.Layers[-1];

            BatchOutputMat losses;
            For_enumerate(losses.Stripes) it = m.Loss.Calculate(targets->Stripes[it_index], ol->Out.Stripes[it_index + 1]);

            auto costMatrix = T(losses);

            auto cost = costMatrix.Stripes[0];
            For(range(1, costMatrix.R)) cost += costMatrix.Stripes[it];
            cost /= (f32)N_TRAIN_EXAMPLES_PER_STEP;

            // cost derivative
            BatchOutputMat d;
            For_enumerate(d.Stripes) it = m.Loss.GetDerivative(targets->Stripes[it_index], ol->Out.Stripes[it_index + 1]);

            void* delta = &d;

            static_for<1, ARCHITECTURE_COUNT>([&](auto i) {
                using T = std::tuple_element_t<ARCHITECTURE_COUNT - i, ARCHITECTURE_T>;
                using TM1 = std::tuple_element_t<ARCHITECTURE_COUNT - i - 1, ARCHITECTURE_T>;

                using L_T = dense_layer_runtime<TM1::NUM_NEURONS, T::NUM_NEURONS, T::Activation>;
                auto l = (L_T*)m.Layers[-i]; // Negative indexing means from the back
                l->calculate_delta(delta);
                delta = &l->DeltaWeighted;
            });

            // Finally, do the weight updates
            For(m.Layers) it->update_weights(m.LearningRate);
        }

        fmt::print("{0}/{0} ", m.Batches.Count * N_TRAIN_EXAMPLES_PER_STEP);
        fmt::print("[{:=<{}}]\n", "", eq_per_batch * m.Batches.Count);
    }
    fmt::print("\n");
}

// We do this for syntactic sugar..
struct fit_model_parameters {
    dyn_mat X, y;
    s64 Epochs = 0;
};

inline void fit(model m, fit_model_parameters params)
{
    assert(params.X.R == params.y.R && "Different number of examples in X and y");

    assert(params.X.C == INPUT_SHAPE && "Bad X shape");
    assert(params.y.C == OUTPUT_NEURONS && "Bad y shape");

    assert(params.Epochs >= 0 && "Invalid number of epochs");

    s64 numExamples = params.X.R;

    //
    // Each step includes _N_TRAIN_EXAMPLES_PER_STEP_. It is strongly recommended that you make sure the number of examples
    // is a multiple of _N_TRAIN_EXAMPLES_PER_STEP_ because the way we have set it up, we can't have variable number of training
    // examples per step. Right now, in case it is not a multiple, the last batch (which would have less than _N_TRAIN_EXAMPLES_PER_STEP_)
    // is discarded. We could also create a new batch and copy some of the existing examples, but that should be left up to the caller!
    //
    // Before dividing into batches we shuffle the input.
    // Later when training we also shuffle the batches, so each step starts with a random batch.
    //
    // Note: For reproducible results, seed the random generator with a custom number.
    // In the examples, at the end of execution we print the seed in any case,
    // so you don't lose it if you need to repeat the same experiment
    //
    s64 stepsPerEpoch = numExamples / N_TRAIN_EXAMPLES_PER_STEP;

    // We generate a list of indices which we then shuffle.
    // That's how we index the examples later.
    array<s64> indices;
    For(range(stepsPerEpoch)) append(indices, it);

    // Shuffle using Fisher–Yates algorithm. Assumption: is this statistically good enough?
    // Right now random numbers are generated using C's rand() library which is NOT good enough for cryptography for e.g.
    // In the future we should consider replacing it.
    shuffle(indices.Data, indices.Data + indices.Count);

    For(range(stepsPerEpoch))
    {
        // Since we are using a custom allocator, these are guaranteed to be next to each other.
        // This is good the CPU cache and may drastically improve performance.
        //
        // Note: When choosing _N_TRAIN_EXAMPLES_PER_STEP_ you may want to consider
        // the cache size of your CPU and make it so one batch can fit neatly. Of course
        // this means that each step will train on less samples and weight updates will be
        // more frequent. When _N_TRAIN_EXAMPLES_PER_STEP_ is 1 we get Stochastic Gradient Descent.
        //
        // @TODO: Can we actually calculate that cache thing with code? Would be a good feature of this library.
        //
        // We first allocate the transposed types because they have the examples in the rows
        // (like we expect them from the user). We later transpose them so they are the in
        // the shape the layers expect them to be in.
        auto batch_in_transposed = new decltype(T(BatchInputMat {}));
        auto batch_out_transposed = new decltype(T(BatchOutputMat {}));

        s64 batch_x_stride = N_TRAIN_EXAMPLES_PER_STEP * INPUT_SHAPE; // How many floats in each batch
        s64 batch_y_stride = N_TRAIN_EXAMPLES_PER_STEP * OUTPUT_NEURONS; // How many floats in each batch

        auto* batch_begin_x = &params.X.Data[indices[it] * batch_x_stride];
        auto* batch_end_x = batch_begin_x + batch_x_stride;

        auto* batch_begin_y = &params.y.Data[indices[it] * batch_y_stride];
        auto* batch_end_y = batch_begin_y + batch_y_stride;

        auto* p = batch_begin_x;
        For(batch_in_transposed->Stripes)
        {
            it.Data[0] = 1.0f; // Augment the input to include a "1" for the bias term
            copy_memory(&it.Data[1], p, INPUT_SHAPE * sizeof(f32));
            p += INPUT_SHAPE;
        }
        assert(p == batch_end_x); // Sanity

        p = batch_begin_y;
        For(batch_out_transposed->Stripes)
        {
            copy_memory(&it.Data[0], p, OUTPUT_NEURONS * sizeof(f32));
            p += OUTPUT_NEURONS;
        }
        assert(p == batch_end_y); // Sanity

        auto* bin = new BatchInputMat;
        *bin = T(*batch_in_transposed);

        auto* bout = new BatchOutputMat;
        *bout = T(*batch_out_transposed);

        append(m.Batches, std::make_tuple(bin, bout));
    }

    train(m, params.Epochs);
}

inline array<vecf<OUTPUT_NEURONS>> predict(model m, array_view<f32> X)
{
    assert(X.Count % INPUT_SHAPE == 0 && "Bad X shape");

    dyn_mat Xdyn;
    append_array(Xdyn.Data, X);
    Xdyn.R = X.Count / INPUT_SHAPE;
    Xdyn.C = INPUT_SHAPE;

    dyn_mat Xt = augment_dyn_mat(Xdyn);

    auto o = T(Xt); // Holds the last output of the layers
    free(Xt.Data);

    static_for<1, ARCHITECTURE_COUNT>([&](auto i) {
        using _T = std::tuple_element_t<i, ARCHITECTURE_T>;
        using _TM1 = std::tuple_element_t<i - 1, ARCHITECTURE_T>;

        using L_T = dense_layer_runtime<_TM1::NUM_NEURONS, _T::NUM_NEURONS, _T::Activation>;

        auto l = (L_T*)m.Layers[i - 1]; // -1 for the first layer (which is the input layer)

        matf<_T::NUM_NEURONS, _TM1::NUM_NEURONS + 1, true> packed_weights = l->Weights;

        dyn_mat weights;
        append_pointer_and_size(weights.Data, &packed_weights.Stripes[0][0], packed_weights.R * packed_weights.C);
        weights.R = packed_weights.R;
        weights.C = packed_weights.C;

        auto new_o = dot(weights, o);
        free(o.Data);

        For(new_o.Data) it = _T::Activation::apply(it); // @Performance

        if constexpr (i != ARCHITECTURE_COUNT - 1) {
            auto tout = T(new_o);
            auto augmented_tout = augment_dyn_mat(tout);

            o = T(augmented_tout);

            free(tout.Data);
            free(augmented_tout.Data);
        } else {
            o = T(new_o);
        }
        free(new_o.Data);
        free(weights.Data);
    });

    array<vecf<OUTPUT_NEURONS>> y;
    reserve(y, X.Count);

    For(range(o.R))
    {
        vecf<OUTPUT_NEURONS> t;
        copy_memory(&t, &o.Data[it * o.C], OUTPUT_NEURONS * sizeof(f32));
        append(y, t);
    }
    return y;
}
