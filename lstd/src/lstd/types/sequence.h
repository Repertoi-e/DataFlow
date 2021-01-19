#pragma once

#include "basic_types.h"

//
// This file provides types to work with templates and a sequence of things (most commonly integral types).
//

LSTD_BEGIN_NAMESPACE

template <typename T_, T_... Elements>
struct sequence {
    using type = sequence;

    using T = T_;
    static constexpr s64 SIZE = sizeof...(Elements);
};

template <typename T, typename S1, typename S2>
struct concat;

template <typename T, T... I1, T... I2>
struct concat<T, sequence<T, I1...>, sequence<T, I2...>> : sequence<T, I1..., (sizeof...(I1) + I2)...> {
};

template <typename T, s64 N>
struct make_sequence_impl : concat<T, typename make_sequence_impl<T, N / 2>::type, typename make_sequence_impl<T, N - N / 2>::type> {
};

template <typename T>
struct make_sequence_impl<T, 0> : sequence<T> {
};

template <typename T>
struct make_sequence_impl<T, 1> : sequence<T, 0> {
};

template <s64... Is>
using integer_sequence = sequence<s64, Is...>;

template <typename T, s64 N>
using make_sequence = typename make_sequence_impl<T, N>::type;

template <s64 N>
using make_integer_sequence = typename make_sequence_impl<s64, N>::type;

template <typename IS1, typename IS2>
struct merge_sequence;

template <typename T, T... Indices1, T... Indices2>
struct merge_sequence<sequence<T, Indices1...>, sequence<T, Indices2...>> {
    using type = sequence<T, Indices1..., Indices2...>;
};

template <typename IS1, typename IS2>
struct merge_integer_sequence;

template <s64... Indices1, s64... Indices2>
struct merge_integer_sequence<integer_sequence<Indices1...>, integer_sequence<Indices2...>> {
    using type = integer_sequence<Indices1..., Indices2...>;
};

template <typename IS>
struct reverse_sequence;

template <typename T, T Head, T... Indices>
struct reverse_sequence<sequence<T, Head, Indices...>> {
    using type = typename merge_sequence<typename reverse_sequence<sequence<T, Indices...>>::type, sequence<T, Head>>::type;
};

template <typename IS>
struct reverse_integer_sequence;

template <s64 Head, s64... Indices>
struct reverse_integer_sequence<integer_sequence<Head, Indices...>> {
    using type = typename merge_integer_sequence<typename reverse_integer_sequence<integer_sequence<Indices...>>::type, integer_sequence<Head>>::type;
};

LSTD_END_NAMESPACE
