// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_PYTHON_PY_TYPES_H_
#define DRAGON_PYTHON_PY_TYPES_H_

#include <numpy/arrayobject.h>
#include "core/types.h"
using namespace dragon;

inline const int TypeMetaToNPY(const TypeMeta& meta) {
    static std::unordered_map<TypeId, int> m2npy_type_map {
        { TypeMeta::Id<float>(), NPY_FLOAT32 },
        { TypeMeta::Id<int>(), NPY_INT32 },
        { TypeMeta::Id<int64_t>(), NPY_INT64 },
        { TypeMeta::Id<double>(), NPY_FLOAT64 },
        { TypeMeta::Id<float16>(), NPY_FLOAT16 },
        { TypeMeta::Id<uint8>(), NPY_UINT8 },
        { TypeMeta::Id<int8>(), NPY_INT8 }
    };
    return m2npy_type_map.count(meta.id()) ? m2npy_type_map[meta.id()] : -1;
}

inline const TypeMeta& TypeNPYToMeta(int npy_type) {
    static std::unordered_map<int, TypeMeta> npy2m_type_map {
        { NPY_FLOAT32, TypeMeta::Make<float>() },
        { NPY_INT32, TypeMeta::Make<int>() },
        { NPY_INT64, TypeMeta::Make<int64_t>() },
        { NPY_FLOAT64, TypeMeta::Make<double>() },
        { NPY_FLOAT16, TypeMeta::Make<float16>() },
        { NPY_UINT8, TypeMeta::Make<uint8>() },
        { NPY_INT8, TypeMeta::Make<int8>() },
    };
    static TypeMeta unknown_type;
    return npy2m_type_map.count(npy_type) ?
        npy2m_type_map[npy_type] : unknown_type;
}

#endif    // DRAGON_PYTHON_PY_TYPES_H_