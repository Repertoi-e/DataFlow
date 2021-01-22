#pragma once

#include <lstd/file.h>
#include <lstd/fmt.h>
#include <lstd/io.h>
#include <lstd/math.h>
#include <lstd/memory/array.h>
#include <lstd/memory/hash_table.h>
#include <lstd/os.h>

#include <time.h>

#include <tuple>

//
// Note for me (Dimitar Sotirov).
// Useful things to include in the original lstd:
//
// The modifications in the math module!!!
//

template <typename T, typename TIter = decltype(std::begin(std::declval<T>())), typename = decltype(std::end(std::declval<T>()))>
constexpr auto enumerate_impl(T&& in)
{
    struct iterator {
        size_t i;
        TIter iter;
        bool operator!=(const iterator& other) const { return iter != other.iter; }
        void operator++()
        {
            ++i;
            ++iter;
        }
        auto operator*() const { return std::tie(i, *iter); }
    };
    struct iterable_wrapper {
        T iterable;
        auto begin() { return iterator { 0, std::begin(iterable) }; }
        auto end() { return iterator { 0, std::end(iterable) }; }
    };
    return iterable_wrapper { std::forward<T>(in) };
}

//
// Inspired from Python's enumerate().
// Example usage:
//
//    For_enumerate(data) {
//        other_data[it_index] = it + 1;
//    }
//
// .. which is the same as:
//
//    For(range(data.Count)) {
//        other_data[it] = data[it] + 1;
//    }
//
// Might not look much shorter but you don't a separate
// variable if you use data[it] more than once.
// It's just a convenience.
//
// You can change the names of the internal
// variables by using _For_enumerate_as_.
//
#define For_enumerate_as(it_index, it, in) for (auto [it_index, it] : enumerate_impl(in))
#define For_enumerate(in) For_enumerate_as(it_index, it, in)

// This template function unrolls a loop at compile-time.
// The lambda should take "auto it" as a parameter and
// that can be used as a compile-time constant index.
//
// This is useful for when you can just write a for-loop 
// instead of using template functional recursive programming.
template <s64 First, s64 Last, typename Lambda>
void static_for(Lambda&& f)
{
    if constexpr (First < Last) {
        f(types::integral_constant<s64, First> {});
        static_for<First + 1, Last>(f);
    }
}

//
// Also the modifications in the math module!!!
//
