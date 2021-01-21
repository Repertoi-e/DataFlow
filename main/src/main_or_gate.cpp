#include "pch.h"

#include "architecture.h"

//
// We define this
//

constexpr s64 N_TRAIN_EXAMPLES_PER_STEP = 1;

constexpr auto ARCHITECTURE = std::make_tuple(
    input<2> {},
    dense<1, activation_sigmoid> {});

//
// This is infered:
//

using ARCHITECTURE_T = decltype(ARCHITECTURE);
constexpr s64 ARCHITECTURE_COUNT = std::tuple_size_v<ARCHITECTURE_T>;

constexpr s64 INPUT_SHAPE = std::tuple_element_t<0, ARCHITECTURE_T>::NUM_NEURONS;
constexpr s64 OUTPUT_NEURONS = std::tuple_element_t<ARCHITECTURE_COUNT - 1, ARCHITECTURE_T>::NUM_NEURONS;

#include "model.h" // Include this after architecture has been defined. Sadly I don't think there is a way around this!

s32 main()
{
    Context.AllocAlignment = 16; // For SIMD

    // To ensure optimal performace we allocate all the memory we need next to
    // each other. We use an arena allocator to do that. If it does not have
    // enough space it falls back to regular malloc. So we need to make sure that
    // we allocate enough memory upfront.
    //
    // After running once we can use: Context.TempAllocData.TotalUsed; to see how
    // much memory we've actually used. Just put a number bigger than that here.
    allocate_array(byte, 1_MiB, Context.Temp);
    free_all(Context.Temp);

    srand((u32)time(NULL));

    // Deterministic initialization
    // srand(42);

    //
    // We attempt to train an OR gate:
    //
    // [0, 0] -> 0
    // [1, 0] -> 1
    // [0, 1] -> 1
    // [1, 1] -> 1
    //
    auto input = to_stack_array<f32>(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f);
    auto targets = to_stack_array<f32>(0.0f, 1.0f, 1.0f, 1.0f);

    dyn_mat X, y;
    append_pointer_and_size(X.Data, input.Data, input.Count);
    append_pointer_and_size(y.Data, targets.Data, targets.Count);
    X.R = input.Count / INPUT_SHAPE;
    X.C = INPUT_SHAPE;
    y.R = targets.Count / OUTPUT_NEURONS;
    y.C = OUTPUT_NEURONS;

    model m;
    WITH_ALLOC(Context.Temp)
    {
        m = compile_model({ .LearningRate = 1.0f, .Loss = BinaryCrossEntropy });
        fit(m, { .X = X, .y = y, .Epochs = 100 });
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

        val_targets[it] = v1(a | b);
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
