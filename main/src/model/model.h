#pragma once

#include "../pch.h"

// Infer these from the architecture which should be defined before including this file...
using ARCHITECTURE_T = decltype(ARCHITECTURE);
constexpr s64 ARCHITECTURE_COUNT = std::tuple_size_v<ARCHITECTURE_T>;

constexpr s64 INPUT_SHAPE = std::tuple_element_t<0, ARCHITECTURE_T>::NUM_NEURONS;
constexpr s64 OUTPUT_NEURONS = std::tuple_element_t<ARCHITECTURE_COUNT - 1, ARCHITECTURE_T>::NUM_NEURONS;

#include "dyn_mat.h"
#include "loss.h"
#include "runtime_layer.h"

struct model {
    array<base_layer_runtime*> Layers;

    f32 LearningRate;
    f32 B1, B2; // Exponential decay rates for the moment estimates (:Adam)

    loss_function Loss;

    array<std::tuple<BatchInputMat*, BatchOutputMat*>> Batches;
};

// This is used for an argument to _compile_model_ for a nice syntactic
// sugar that makes it obvious which parameter we are setting.
struct hyper_parameters {
    f32 LearningRate;
    f32 B1, B2; // Exponential decay rates for the moment estimates (:Adam)

    loss_function Loss;
};

// This returns a new model each time:
// - Allocates the runtime layers using the specified architecture.
// - Initializes weights (currently there is no way to do custom weight-initialization @TODO!)
// - Sets up the specified hyper-parameters.
inline model compile_model(hyper_parameters h_params)
{
    model result;

    auto layers = get_layers_from_architecture();
    result.Layers = layers;

    // Weight initialization. For now we generate random numbers between -1 and 1.
    // In the future we should definitely support different initializaton methods!
    static_for<1, ARCHITECTURE_COUNT>([&](auto it) {
        auto* l = (DLR_T<it>*)result.Layers[it];

        For_as(stripe, l->Weights.Stripes)
        {
            // This is ugly.
            // Also I'm not sure if rand() is statistically good enough.
            For(stripe) it = ((f32)rand() / (32767 / 2)) - 1;
            stripe[0] = 0.0f; // Init bias to zero
        }

        // :Adam
        l->FirstRawMoment = 0.0f;
        l->SecondRawMoment = 0.0f;
    });

    // Initialize the rest of hyper-parameters
    assert(h_params.Loss.Calculate != null && "Forgot loss function");
    result.Loss = h_params.Loss;
    result.LearningRate = h_params.LearningRate;
    result.B1 = h_params.B1;
    result.B2 = h_params.B2;

    return result;
}

// _I_ is the starting layer index. Since input is at 0, this is normally 1.
//
// :DontCircumventTypechecking
// We can't directly static_for this because the input of a layer is the output of the previous layer.
// In order to not circumvent typecheking we use a recursive template function. When the code is actually
// compiled this is unrolled (the result is every layer operating inlined directly one after the other),
// so the compiler has extra opportunity for optimization.
template <s64 I>
auto feedforward(model m, const decltype(DLR_T<I - 1>::Out)& in)
{
    auto* l = (DLR_T<I>*)m.Layers[I];

    l->Out.get_row_view<l->Out.R - 1>(1) = DLR_T<I>::Activation::apply(dot(l->Weights, in));
    l->Out.row(0) = vecf<N_TRAIN_EXAMPLES_PER_STEP>(1.0f); // Augment the output to account for the bias term

    if constexpr (I < ARCHITECTURE_COUNT - 1) {
        return feedforward<I + 1>(m, l->Out);
    } else {
        return l->Out.get_row_view<l->Out.R - 1>(1); // We return the output without the bias
    }
}

// We are going backwards, so _next_layer_delta_weighted_ is calculated before this layer's delta.
// _I_ is the starting layer index. Since we are going backwards this is normally ARCHITECTURE_COUNT - 1.
//
// :DontCircumventTypechecking
// We can't directly static_for this because we need the weighted delta of the next layer.
// See note in feedforward(..).
template <s64 I>
void backpropagation(model m, const decltype(DLR_T<I + 1>::DeltaWeighted)& next_layer_delta_weighted)
{
    auto* l = (DLR_T<I>*)m.Layers[I];

    // If this is the final layer _next_layer_delta_weighted_ points to the derivative
    // of the loss function, otherwise it points to next layer's _DeltaWeighted_.

    decltype(l->Delta) out_no_bias = l->Out.get_row_view<l->Out.R - 1>(1);
    l->Delta = next_layer_delta_weighted * DLR_T<I>::Activation::get_derivative(out_no_bias);

    auto delta_weighted = dot(T(l->Weights), l->Delta);
    l->DeltaWeighted = delta_weighted.get_row_view<delta_weighted.R - 1>(1);

    // Input is at layer 0, we stop at layer 1.
    if constexpr (I > 1) {
        backpropagation<I - 1>(m, l->DeltaWeighted);
    }
}

// _I_ is the starting layer index. Since input is at 0, this is normally 1.
//
// _t_ is the current time-step (:Adam)
//
// :DontCircumventTypechecking
// We can't directly static_for this because we need the input of the previous layer.
// See note in feedforward(..).
template <s64 I>
void update_weights(model m, s64 t, const decltype(DLR_T<I - 1>::Out)& in)
{
    auto* l = (DLR_T<I>*)m.Layers[I];

    auto average_weight_update = dot(l->Delta, T(in)) / (f32)N_TRAIN_EXAMPLES_PER_STEP;

    // :Adam
    l->FirstRawMoment = m.B1 * l->FirstRawMoment + (1 - m.B1) * average_weight_update;
    l->SecondRawMoment = m.B2 * l->SecondRawMoment + (1 - m.B2) * (average_weight_update * average_weight_update);

    // We do this bias correction because the first time-steps don't have a moment
    auto first_moment_bias_corrected = l->FirstRawMoment / (1 - pow(m.B1, t));
    auto second_moment_bias_corrected = l->SecondRawMoment / (1 - pow(m.B2, t));

    // +1e-07F to avoid division by zero
    auto coeff = m.LearningRate * first_moment_bias_corrected / (sqrt(second_moment_bias_corrected) + 1e-07F);

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    l->Weights -= m.LearningRate * average_weight_update;

    if constexpr (I < ARCHITECTURE_COUNT - 1) {
        update_weights<I + 1>(m, t, l->Out);
    }
}

// This is called by _fit()_.
// Trains the model for a number of epochs given the already calculated batches.
inline void train(model m, s64 epochs)
{
    For_as(epoch, range(epochs))
    {
        fmt::print("Epoch {}/{}\n", epoch + 1, epochs);

        decltype(T(BatchOutputMat {}))::StripeVecT totalCost;

        f32 eq_per_batch = 40.0f / m.Batches.Count;
        For_enumerate_as(batch_index, batch, m.Batches)
        {
            fmt::print("{}/{} ", batch_index * N_TRAIN_EXAMPLES_PER_STEP, m.Batches.Count * N_TRAIN_EXAMPLES_PER_STEP);
            fmt::print("[{:=<{}}>{:.<{}}]\r", "", (s64)(batch_index * eq_per_batch), "", (s64)max(eq_per_batch * m.Batches.Count - batch_index * eq_per_batch - 1, 0.0f));

            auto [attributes, targets] = batch;

            auto out = feedforward<1>(m, *attributes);

            // Calculate losses
            BatchOutputMat losses = m.Loss.Calculate(*targets, out);

            // Calculate cost
            auto costMatrix = T(losses);

            auto cost = costMatrix.Stripes[0];
            For(range(1, costMatrix.R)) cost += costMatrix.Stripes[it];
            cost /= N_TRAIN_EXAMPLES_PER_STEP;

            totalCost += cost / m.Batches.Count;

            // Calculate cost derivative
            BatchOutputMat d = m.Loss.GetDerivative(*targets, out);

            backpropagation<ARCHITECTURE_COUNT - 1>(m, d);

            s64 t = 1 + batch_index + epoch * m.Batches.Count;
            update_weights<1>(m, t, *attributes);
        }
        fmt::print("{0}/{0} ", m.Batches.Count * N_TRAIN_EXAMPLES_PER_STEP);
        fmt::print("[{:=<{}}] cost: {}\n", "", 40, totalCost);
    }
    fmt::print("\n");
}

// We do this for syntactic sugar..
struct fit_model_parameters {
    dyn_mat X, y;
    s64 Epochs = 0;
};

// Takes X and y and splits them into batches.
// Each batch has _N_TRAIN_EXAMPLES_PER_STEP_ elements (see note in this function's body).
// Also calls _train()_ with the specified number of epochs.
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

// Takes X examples of whatever length and uses the trained model weights to get an output.
// This function currently has a very ugly implementation since we need to deal with dynamic matrices.
// We don't yet have a robust type that can handle e.g. reshaping on the fly (the _fit()_ portion of
// this library works with compile-time parameters).
//
// Note: In the future I might just give up on this since this library is meant to work side by side
// with Python (that's why I have just hacked this thing and haven't given it enough attention yet).
// We can just give the weights to numpy and deploy the model that way. This library's sole purpose
// is just to speed up the training process of the model.
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
        auto l = (DLR_T<i>*)m.Layers[i];
        matf<DLR_T<i>::NUM_NEURONS, DLR_T<i>::NUM_INPUTS + 1, true> packed_weights = l->Weights;

        dyn_mat weights;
        append_pointer_and_size(weights.Data, &packed_weights.Stripes[0][0], packed_weights.R * packed_weights.C);
        weights.R = packed_weights.R;
        weights.C = packed_weights.C;

        auto new_o = dot(weights, o);
        free(o.Data);

        // This fails for softmax.
        For(new_o.Data) it = DLR_T<i>::Activation::apply(it); // @Performance

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
