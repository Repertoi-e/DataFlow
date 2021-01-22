#include "pch.h"

#include "architecture.h"

//
// We define this
//

constexpr s64 N_TRAIN_EXAMPLES_PER_STEP = 100;

constexpr auto ARCHITECTURE = std::make_tuple(
    input<28 * 28> {},
    dense<128, activation_sigmoid> {},
    dense<64, activation_sigmoid> {},
    dense<32, activation_sigmoid> {},
    dense<10, activation_softmax> {});

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
    allocate_array(byte, 300_MiB, Context.Temp);
    free_all(Context.Temp);

    string dataFolder = path::join(os_get_working_dir(), "data");

    auto trainImagesFile = file::handle(path::join(dataFolder, "train-images.idx3-ubyte"));
    auto trainLabelsFile = file::handle(path::join(dataFolder, "train-labels.idx1-ubyte"));

    WITH_ALLOC(Context.Temp)
    {
        s64 numItems;

        // Here we read the file, transform into floats, and scale all in one go.
        array<f32> images;
        {
            auto [trainImagesBytes, success] = trainImagesFile.read_entire_file();
            assert(success);

            defer(free(trainImagesBytes));

            auto* p = trainImagesBytes.Data;
            assert(p[0] == 0 && p[1] == 0 && p[2] == 8 && p[3] == 3); // Magic number 0x00000803
            p += 4;

            numItems = p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3];
            p += 4;

            p += 4 + 4; // the file includes the number of rows and columns

            reserve(images, numItems * INPUT_SHAPE);
            images.Count = numItems * INPUT_SHAPE; // @Dirty!

            For(range(numItems))
            {
                For_as(r, range(28)) For_as(c, range(28)) images[it * INPUT_SHAPE + (r * 28 + c)] = (f32)(p[r * 28 + c]) / 255.0f;
                p += INPUT_SHAPE;
            }
        }

        // Here we read the file, transform into floats, and do one-hot encoding in one go.
        array<f32> labels;
        {
            auto [trainLabelsBytes, success] = trainLabelsFile.read_entire_file();
            assert(success);

            defer(free(trainLabelsBytes));

            auto* p = trainLabelsBytes.Data;
            assert(p[0] == 0 && p[1] == 0 && p[2] == 8 && p[3] == 1); // Magic number 0x00000801
            p += 4;

            s32 numItems2 = p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3];
            assert(numItems2 == numItems); // Sanity!
            p += 4;

            reserve(labels, numItems * OUTPUT_NEURONS);
            labels.Count = numItems * OUTPUT_NEURONS; // @Dirty!

            For(range(numItems))
            {
                auto oneHot = vecf<OUTPUT_NEURONS>(0.0f);
                oneHot[*p++] = 1;
                copy_memory(&labels[it * OUTPUT_NEURONS], &oneHot, OUTPUT_NEURONS * sizeof(f32));
            }
        }

        // srand((u32)time(NULL));

        // Deterministic initialization
        srand(42);

        model m = compile_model({ .LearningRate = 5.0f, .Loss = BinaryCrossEntropy });

        // @TODO Lot's of boilerplate when working with different types in C++. We should make this easier!
        dyn_mat X, y;
        X.Data = images;
        X.R = numItems;
        X.C = INPUT_SHAPE;

        y.Data = labels;
        y.R = numItems;
        y.C = OUTPUT_NEURONS;

        fit(m, { .X = X, .y = y, .Epochs = 200 });

        fmt::print("{} images, {} labels, {} bytes memory used\n", images.Count, labels.Count, Context.TempAllocData.TotalUsed);
    }

    return 0;
}
