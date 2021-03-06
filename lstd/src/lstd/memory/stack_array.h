#pragma once

#include <stdlib.h> // @DependencyCleanup rand()

#include "../internal/common.h"
#include "../types/sequence.h"
#include "array_like.h"
#include "string_utils.h"

LSTD_BEGIN_NAMESPACE

template <typename T>
constexpr T *quick_sort_partition(T *first, T *last, T *pivot) {
    --last;
    swap(*pivot, *last);
    pivot = last;

    while (true) {
        while (*first < *pivot) ++first;
        --last;
        while (*pivot < *last) --last;
        if (first >= last) {
            swap(*pivot, *first);
            return first;
        }
        swap(*first, *last);
        ++first;
    }
}

template <typename T>
constexpr void quick_sort(T *first, T *last) {
    if (first >= last) return;

    auto *pivot = first + (last - first) / 2;
    auto *nextPivot = quick_sort_partition(first, last, pivot);
    quick_sort(first, nextPivot);
    quick_sort(nextPivot + 1, last);
}

// An implementation of Fisher–Yates shuffle algorithm.
// 
// Uses rand() which is seeded by srand().
// @TODO @Robustness In the future we should write a system for random number generators.
template <typename T>
constexpr void shuffle(T* first, T* last)
{
    s64 n = last - first;
    For(range(n - 1, -1, -1)) {
        // Pick a random index from 0 to i
        s64 j = rand() % (it + 1);
     
        // Swap 
        T temp = first[it];
        first[it] = first[j];
        first[j] = temp;
    }
}

template <typename T>
struct array_view;

// @TODO: Document use cases for this and why is it different from array<T>
template <typename T_, s64 N>
struct stack_array {
    using T = T_;

    // :CodeReusability: Automatically generates ==, !=, <, <=, >, >=, compare_*, find_*, has functions etc.. take a look at "array_like.h"
    static constexpr bool IS_ARRAY_LIKE = true;

    T Data[N ? N : 1];
    static constexpr s64 Count = N;

    //
    // Iterators:
    //
    using iterator = T *;
    using const_iterator = const T *;

    constexpr iterator begin() { return Data; }
    constexpr iterator end() { return Data + Count; }
    constexpr const_iterator begin() const { return Data; }
    constexpr const_iterator end() const { return Data + Count; }

    //
    // Operators:
    //
    constexpr operator array_view<T>() const;

    constexpr T &operator[](s64 index) { return Data[translate_index(index, Count)]; }
    constexpr const T &operator[](s64 index) const { return Data[translate_index(index, Count)]; }
};

namespace internal {
template <typename D, typename...>
struct return_type_helper {
    using type = D;
};
template <typename... Types>
struct return_type_helper<void, Types...> : types::common_type<Types...> {};

template <class T, s64 N, s64... I>
constexpr stack_array<types::remove_cv_t<T>, N> to_array_impl(T (&a)[N], integer_sequence<I...>) {
    return {{a[I]...}};
}
}  // namespace internal

template <typename D = void, class... Types>
constexpr stack_array<typename internal::return_type_helper<D, Types...>::type, sizeof...(Types)> to_stack_array(Types &&... t) {
    return {(Types &&)(t)...};
}

template <typename T, s64 N>
constexpr stack_array<types::remove_cv_t<T>, N> to_stack_array(T (&a)[N]) {
    return internal::to_array_impl(a, make_integer_sequence<N>{});
}

LSTD_END_NAMESPACE
