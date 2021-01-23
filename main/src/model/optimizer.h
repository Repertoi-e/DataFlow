#pragma once

#include "pch.h"

//
// Gradient descent (with momentum) optimizer.
//
// Hyper-parameters: LearningRate, Momentum, Nesterov
//
// By default:
//     w = w - LearningRate * g
//
// When Momentum hyper-parameter is not 0:
//     velocity = Momentum * velocity - LearningRate * g
//     w = w + velocity
//
// When Nesterov Momentum is used:
//     velocity = Momentum * velocity - LearningRate * g
//     w = w + Momentum * velocity - LearningRate * g
//
struct SGD {
    f32 LearningRate;
    f32 Momentum = 0.0f;
    bool Nesterov = false;

    struct params {
        f32 LearningRate = 0.001f;
        f32 Momentum = 0.0f;
        bool Nesterov = false;
    };

    SGD(params params = {})
        : LearningRate(params.LearningRate)
        , Momentum(params.Momentum)
        , Nesterov(params.Nesterov)
    {
    }

    template <typename Layer>
    void update_weights(Layer* l, const decltype(Layer::Weights)& gradients)
    {
        if (Momentum == 0.0f) {
            l->Weights -= LearningRate * gradients;
        } else {
            auto& velocity = l->FirstRawMoment;

            velocity = Momentum * velocity - LearningRate * gradients;
            if (!Nesterov) {
                l->Weights += velocity;
            } else {
                l->Weights += Momentum * velocity - LearningRate * gradients;
            }
        }
    }
};

//
// Adam (adaptive moment) optimizer.
//
// Hyper-parameters:
//  - LearningRate
//  - B1, B2 - [0, 1) - Exponential decay rates for the moment estimates
//
// Algorithm for updating the weights:
//
//     first_raw_moment = B1 * first_raw_moment + (1 - B1) * gradients;
//     second_raw_moment = B2 * second_raw_moment + (1 - B2) * (gradients * gradients);
//
//     first_moment_bias_corrected = first_raw_moment / (1.0f - pow(B1, t));
//     second_moment_bias_corrected = second_raw_moment / (1.0f - pow(B2, t));
//
//     w = w - LearningRate * first_moment_bias_corrected / (sqrt(second_moment_bias_corrected) + 1e-07F)
//
struct Adam {
    f32 LearningRate;
    f32 B1, B2;

    struct params {
        f32 LearningRate = 0.001f;
        f32 B1 = 0.9f, B2 = 0.999f;
    };

    Adam(params params = {})
        : LearningRate(params.LearningRate)
        , B1(params.B1)
        , B2(params.B2)
    {
    }

    template <typename Layer>
    void update_weights(Layer* l, const decltype(Layer::Weights)& gradients)
    {
        l->FirstRawMoment = B1 * l->FirstRawMoment + (1 - B1) * gradients;
        l->SecondRawMoment = B2 * l->SecondRawMoment + (1 - B2) * (gradients * gradients);

        // We do this bias correction because the first time-steps don't have a moment
        auto first_moment_bias_corrected = l->FirstRawMoment / (1.0f - pow(B1, l->TimeStep));
        auto second_moment_bias_corrected = l->SecondRawMoment / (1.0f - pow(B2, l->TimeStep));

        // +1e-07F to avoid division by zero
        l->Weights -= LearningRate * first_moment_bias_corrected / (sqrt(second_moment_bias_corrected) + 1e-07F);

        ++l->TimeStep;
    }
};