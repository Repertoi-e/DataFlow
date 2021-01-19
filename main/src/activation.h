#pragma once

template <s64 Dim>
struct activation {
    // Applies the activation function to a single value.
    virtual f32 apply(f32 x) = 0;

    virtual void apply(vecf<Dim>& v)
    {
        For(v) it = apply(it); // @Performance @Math
    }

    template <s64 R, s64 C>
    void apply(mat_view<f32, R, C>& m)
    {
        For_as(i, range(R)) For_as(j, range(C)) m(i, j) = apply(m(i, j)); // @Performance @Math
    }

    using mat_out_t = matf<Dim + 1, N_TRAIN_EXAMPLES_PER_STEP>;
    using mat_d_t = matf<Dim, N_TRAIN_EXAMPLES_PER_STEP>;

    virtual mat_d_t get_derivative(const mat_out_t& t) = 0;
};

template <s64 NumNeurons>
struct activation_func_none : activation<NumNeurons> {
    f32 apply(f32 x) override { return x; }

    // Derivative of identity function is 1
    activation::mat_d_t get_derivative(const activation::mat_out_t& x) override
    {
        return activation::mat_d_t(1.0f);
    }
};

template <s64 NumNeurons>
struct activation_func_sigmoid : activation<NumNeurons> {
    f32 apply_single(f32 x) override { return 1 / (1 + expf(-x)); }

    // Derivative of sigmoid is s(x)(1 - s(x))
    activation::mat_d_t get_derivative(const activation::mat_out_t& x) override
    {
        return (x * (activation::mat_out_t(1.0f) - x)).get_view<NumNeurons, N_TRAIN_EXAMPLES_PER_STEP>(1, 0);
    }
};

template <s64 NumNeurons>
struct activation_func_relu : activation<NumNeurons> {
    // @Performance We can optimize this by looking at the sign bit
    f32 apply_single(f32 x) override { return max(0, x); }

    // Derivative of RELU is 1 when x is non-negative and 0 otherwise
    activation::mat_d_t get_derivative(const activation::mat_out_t& x) override
    {
        activation::mat_d_t result = { no_init };
        For(range(1, NumNeurons))
        {
            // @Performance We can optimize this by looking at the sign bit
            For_enumerate_as(x_index, xx, x.Stripes[it])
            {
                result(it - 1, x_index) = (f32)(xx >= 0);
            }
        }
        return result;
    }
};

template <s64 NumNeurons>
struct activation_func_softmax : activation<NumNeurons> {
    f32 apply_single(f32 x) override { assert(false && "Cannot apply softmax to a single value"); }

    virtual void apply(vecf<Dim>& v)
    {
        auto e = element_wise_exp(v - element_wise_max(v));
        return e / sum(e);
    }

    // Derivative of softmax is s(x)(1 - s(x))
    activation::mat_d_t get_derivative(const activation::mat_out_t& x) override
    {
        // auto result = (x * (activation::mat_out_t(1.0f) - x)).get_view<NumNeurons, N_TRAIN_EXAMPLES_PER_STEP>(1, 0);
    }
};
