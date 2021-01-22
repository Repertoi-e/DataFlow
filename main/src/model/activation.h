#pragma once

#include "../pch.h"

struct activation_none {
    // Applies activation function to a single value, a vector or a matrix.
    // The expression in _apply_ should compile under any of these three types.
    //
    // My library _lstd_ aims to allow this type of Pythonic code expressivity
    // in a strongly typed language like C++. Additions which improve on that
    // are very welcome and API is not yet totally fixed.
    //
    // .. In any case if you need to check for different types you can just
    // create overloads of the _apply_ function taking specialized arguments
    // (e.g. just vec or mat)
    template <typename T>
    static T apply(const T& x) { return x; }

    // Gets the derivative of the activation function given the neuron outputs _a_.
    // These should not include the bias term.
    template <s64 R, s64 C>
    static matf<R, C> get_derivative(const matf<R, C>& a) { return 1.0f; } // Derivative of the identify function is 1.
};

struct activation_sigmoid {
    template <typename T>
    static T apply(const T& x) { return 1.0f / (1.0f + exp(-x)); }

    template <s64 R, s64 C>
    static matf<R, C> get_derivative(const matf<R, C>& a) { return (a * (1.0f - a)); } // Derivative of sigmoid is s(x)(1 - s(x))
};

struct activation_relu {
    template <typename T>
    static T apply(const T& x) { return max(x - 1e-07F, T(0.0f)); } // @Performance We can optimize this by looking at the sign bit

    template <s64 R, s64 C>
    static matf<R, C> get_derivative(const matf<R, C>& a)
    {
        auto result = max(a - 1e-07F, matf<R, C>(0.0f)) / (a - 1e-07F);
        return result;
    } // 0 when x < 0, and 1 otherwise
};

struct activation_softmax {
    static f32 apply(f32 x) { assert(false && "Cannot apply softmax on a single value."); }

    template <s64 Dim>
    static vecf<Dim> apply(const vecf<Dim>& v)
    {
        auto e = exp(v - max(v)); // We subtract max(v) to be more numerically stable
        return e / sum(e);
    }

    template <s64 R, s64 C>
    static matf<R, C> apply(const matf<R, C>& m)
    {
        // @Performance .. my eyes :(
        // Look. Until activation functions become a performance concern
        // it's not worth the effort (and code unreadibility) to optimize it.

        // In this case, since the matrices our layers deal with have the shape
        // NumNeurons x N_TRAIN_EXAMPLES_PER_STEP, i.e. each row actually
        // includes the neuron value for that specific training example,
        // we need to transpose it so each row includes the neurons for a training
        // example. After that we apply softmax to each row-vector.
        //
        // I'm saying "row" but in the library they are refered to as Stripes,
        // because we can switch to column-major representation very easily.
        auto t = T(m);
        For(t.Stripes) apply(it);
        return T(t);
    }

    template <s64 R, s64 C>
    static matf<R, C> get_derivative(const matf<R, C>& a)
    {
        return a;
    }
};
