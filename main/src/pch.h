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

#define For_enumerate_as(it_index, it, in) for (auto [it_index, it] : enumerate_impl(in))
#define For_enumerate(in) For_enumerate_as(it_index, it, in)
