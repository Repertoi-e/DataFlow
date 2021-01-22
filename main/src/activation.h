#pragma once

// We provide so much overloads ... is this needed?

struct activation_none {
    // Applies the activation function to a single value.
    static f32 apply(f32 x) { return x; }

    // Applies the activation function (in place) to a vector of arbitrary dimension.
    // This is usually a set of neurons.
    template <s64 Dim>
    static void apply(vecf<Dim>& v) { return; }

    // Applies the activation function (in place) to a matrix of arbitrary dimension.
    // This is usually the _Out_ matrix with _R_ neurons and _C_ training samples.
    template <s64 R, s64 C>
    static void apply(matf<R, C>& m) { return; }

    // Gets the derivative of the activation function given the neuron outputs _a_.
    // These should not include the bias term.
    //
    // Derivative of the identify function is 1.
    template <s64 R, s64 C>
    static matf<R, C> get_derivative(const matf<R, C>& a) { return matf<R, C>(1.0f); }
};

struct activation_sigmoid {
    static f32 apply(f32 x) { return 1.0f / (1.0f + expf(-x)); }

    template <s64 Dim>
    static void apply(vecf<Dim>& v) { For(v) it = apply(it); }

    template <s64 R, s64 C>
    static void apply(matf<R, C>& m) { For(m.Stripes) apply(it); }

    // Derivative of sigmoid is s(x)(1 - s(x))
    template <s64 R, s64 C>
    static matf<R, C> get_derivative(const matf<R, C>& a) { return (a * (matf<R, C>(1.0f) - a)); }
};

// @Performance We can optimize this by looking at the sign bit
struct activation_relu {
    static f32 apply(f32 x) { return max(0, x); }

    template <s64 Dim>
    static void apply(vecf<Dim>& v) { For(v) it = max(0, it); }

    template <s64 R, s64 C>
    static void apply(matf<R, C>& m) { For(m.Stripes) apply(it); }

    // Derivative of sigmoid is s(x)(1 - s(x))
    template <s64 R, s64 C>
    static matf<R, C> get_derivative(const matf<R, C>& a)
    {
        matf<R, C> result = a;
        For(result.Stripes) it = max(0, it) / it;
        return result;
    }
};

struct activation_softmax {
    static f32 apply(f32 x) { assert(false && "Cannot apply softmax on a single value. This is probably a mistake."); }

    template <s64 Dim>
    static void apply(vecf<Dim>& v)
    {
        // We -max(v) to be more numerically stable
        auto e = element_wise_exp(v - max(v));
        v = e / sum(e);
    }

    template <s64 R, s64 C>
    static void apply(matf<R, C>& m)
    {
        // @Performance My eyes...
        auto t = T(m);
        For(t.Stripes) apply(it);
        m = T(t);
    }

    // Derivative of sigmoid is s(x)(1 - s(x))
    template <s64 R, s64 C>
    static matf<R, C> get_derivative(const matf<R, C>& a)
    {
        return a;
    }
};
