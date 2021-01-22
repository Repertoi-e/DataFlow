#pragma once

#include "../pch.h"

//
// This file includes a stripped down rushed implementation of a matrix that supports any dimension during run-time.
// We use this when using _predict(model, ..)_ for e.g. because we want to support any count of input attributes and
// not do them one by one. In the future we will have a way more elaborate implementation but for now this barely works.
// In the future we will include this in the math module in lstd, because it's a useful thing to have.
//

struct dyn_mat {
    s64 R, C;
    array<f32> Data;
};

inline dyn_mat T(dyn_mat m)
{
    dyn_mat result;
    reserve(result.Data, m.R * m.C);
    result.Data.Count = m.R * m.C; // We init later

    result.R = m.C;
    result.C = m.R;

    For_as(r, range(m.R))
    {
        For_as(c, range(m.C))
        {
            result.Data[c * result.C + r] = m.Data[r * m.C + c];
        }
    }
    return result;
}

inline dyn_mat dot(dyn_mat a, dyn_mat b)
{
    dyn_mat result;
    assert(a.C == b.R);
    reserve(result.Data, a.R * b.C);

    result.R = a.R;
    result.C = b.C;

    result.Data.Count = a.R * b.C;
    fill_memory(result.Data.Data, 0, a.R * b.C * sizeof(f32));

    For_as(i, range(a.R))
    {
        For_as(j, range(b.C))
        {
            For_as(k, range(a.C))
            {
                result.Data[i * b.C + j] += a.Data[i * a.C + k] * b.Data[k * b.C + j];
            }
        }
    }
    return result;
}

// Augment the input to account for the bias
inline dyn_mat augment_dyn_mat(dyn_mat m)
{
    dyn_mat a;
    a.R = m.R;
    a.C = m.C + 1;

    reserve(a.Data, a.R * a.C);
    a.Data.Count = a.R * a.C; // We init later

    auto* p = &a.Data.Data[0];
    auto* t = &m.Data.Data[0];

    For(range(m.R))
    {
        *p = 1.0f;
        ++p;
        copy_memory(p, t, m.C * sizeof(f32));
        p += m.C, t += m.C;
    }
    return a;
}
