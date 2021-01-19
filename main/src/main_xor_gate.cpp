#include "pch.h"

#define INPUT_SHAPE 2
#define OUTPUT_NEURONS 1
#define N_TRAIN_EXAMPLES_PER_STEP 4

#include "model.h" // Include after #defines above!

auto get_layers()
{
    // We can't train an XOR gate without hidden layers because the function is not linearly separable!
    // constexpr auto architecture = dense_layer_architecture<INPUT_SHAPE, OUTPUT_NEURONS>();
    constexpr auto architecture = dense_layer_architecture<INPUT_SHAPE, 2, OUTPUT_NEURONS>();

    array<base_layer*> layers;
    append(layers, new dense_layer<architecture[0]>(ActivationSigmoid));
    append(layers, new dense_layer<architecture[1]>(ActivationSigmoid));
    return layers;
}

s32 main()
{
    // To ensure optimal performace we allocate all the memory we need next to
    // each other. We use an arena allocator to do that. If it does not have
    // enough space it falls back to regular malloc. So we need to make sure that
    // we allocate enough memory upfront.
    //
    // After running once we can use: Context.TempAllocData.TotalUsed; to see how
    // much memory we've actually used. Just put a number bigger than that here.
    allocate_array(byte, 1_MiB, Context.Temp);
    free_all(Context.Temp);

    Context.AllocAlignment = 16; // For SIMD

    srand((u32)time(NULL));

    // Deterministic initialization
    // srand(42);

    //
    // We attempt to train an XOR gate:
    //
    // [0, 0] -> 0
    // [1, 0] -> 1
    // [0, 1] -> 1
    // [1, 1] -> 0
    //
    auto input = to_stack_array<f32>(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f);
    auto targets = to_stack_array<f32>(0.0f, 1.0f, 1.0f, 0.0f);

    model m;

    WITH_ALLOC(Context.Temp)
    {
        m = compile_model({ .Layers = get_layers(), .LearningRate = 5.0f, .Loss = BinaryCrossEntropy });
        fit_model(m, { .X = input, .y = targets, .Epochs = 500 });
    }

    // Here we generate random validation data..
    const s64 VAL_SAMPLES = 50;

    stack_array<f32, VAL_SAMPLES * INPUT_SHAPE> val_input;
    stack_array<vecf<OUTPUT_NEURONS>, VAL_SAMPLES> val_targets;

    For(range(VAL_SAMPLES))
    {
        s32 a = rand() % 2, b = rand() % 2;

        val_input[it * 2] = (f32)a;
        val_input[it * 2 + 1] = (f32)b;

        val_targets[it] = v1(a ^ b);
    }

    auto predicted = predict(m, val_input);

    s64 score = 0;
    For_enumerate(predicted)
    {
        For_as(x, it) x = (x > 0.5f) ? 1.0f : 0.0f;
        if (it == val_targets[it_index]) {
            score += 1;
        }
    }
    fmt::print("{}, accuracy: {}\n", predicted, (f32)score / VAL_SAMPLES);

    fmt::print("{} bytes memory used\n", Context.TempAllocData.TotalUsed);

    return 0;
}
