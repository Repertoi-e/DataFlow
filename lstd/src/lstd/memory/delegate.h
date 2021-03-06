#pragma once

#include "../internal/common.h"
#include "array_like.h"

LSTD_BEGIN_NAMESPACE

//
// This file is a modified implementation of a delegate by Vadim Karkhin.
// https://github.com/tarigo/delegate
//
// Here is the license that came with it:
//
// The MIT License (MIT)
//
// Copyright (c) 2015 Vadim Karkhin
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

template <typename T>
struct delegate;

// This is an object which can store a global function, a method (binded to some instance), or a functor / lambda.
// It doesn't allocate any dynamic memory.
template <typename R, typename... A>
struct delegate<R(A...)> {
    using stub_t = R (*)(void *, A &&...);
    using return_t = R;

    template <typename Type, typename Signature>
    struct target {
        Type *InstancePtr;
        Signature FunctionPtr;
    };

    struct default_type_;                                          // Unknown default type (undefined)
    using default_function = void (default_type_::*)(void);        // Unknown default function (undefined)
    using default_type = target<default_type_, default_function>;  // Default target type

    // :CodeReusability: Automatically generates ==, !=, <, <=, >, >=, compare_*, find_*, has functions etc.. take a look at "array_like.h"
    static constexpr bool IS_ARRAY_LIKE = true;

    static constexpr s64 Count = sizeof(default_type);

    alignas(default_type) byte Data[Count]{};
    alignas(stub_t) stub_t Invoker = null;

    // Invoke static method / free function
    template <decltype(null), typename Signature>
    static R invoke(void *data, A &&... args) {
        return (*reinterpret_cast<const target<decltype(null), Signature> *>(data)->FunctionPtr)((A &&)(args)...);
    }

    // Invoke method
    template <typename Type, typename Signature>
    static R invoke(void *data, A &&... args) {
        return (reinterpret_cast<const target<Type, Signature> *>(data)->InstancePtr->*reinterpret_cast<const target<Type, Signature> *>(data)->FunctionPtr)((A &&)(args)...);
    }

    // Invoke function object (functor)
    template <typename Type, decltype(null)>
    static R invoke(void *data, A &&... args) {
        return (*reinterpret_cast<const target<Type, decltype(null)> *>(data)->InstancePtr)((A &&)(args)...);
    }

    delegate() {}

    // Construct from null
    delegate(decltype(null)) {}

    // Construct delegate with static method / free function
    delegate(R (*function)(A...)) {
        using Signature = decltype(function);

        auto storage = (target<decltype(null), Signature> *) &Data[0];
        storage->InstancePtr = null;
        storage->FunctionPtr = function;
        Invoker = &delegate::invoke<null, Signature>;
    }

    // Construct delegate with method
    template <typename Type, typename Signature>
    delegate(Type *object, Signature method) {
        auto storage = (target<Type, Signature> *) &Data[0];
        storage->InstancePtr = object;
        storage->FunctionPtr = method;
        Invoker = &delegate::invoke<Type, Signature>;
    }

    // using nullptr_t = decltype(null);

    // Construct delegate with function object (functor) / lambda
    template <typename Type>
    delegate(Type *functor) {
        auto storage = (target<Type, nullptr_t> *) &Data[0];
        storage->InstancePtr = functor;
        storage->FunctionPtr = null;
        Invoker = &delegate::invoke<Type, null>;
    }

    // Assign null pointer
    delegate &operator=(decltype(null)) {
        zero_memory(Data, Count);
        Invoker = null;
        return *this;
    }

    bool operator==(decltype(null)) const { return !Invoker; }
    bool operator!=(decltype(null)) const { return Invoker; }

    operator bool() const { return Invoker; }

    // Call delegate
    R operator()(A... args) const {
        return (*Invoker)((void *) &Data[0], (A &&)(args)...);
    }
};

LSTD_END_NAMESPACE
