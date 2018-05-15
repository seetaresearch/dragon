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

#ifndef DRAGON_CORE_TYPES_H_
#define DRAGON_CORE_TYPES_H_

#include <unordered_map>

#include "core/typeid.h"

namespace dragon {

#ifdef _MSC_VER

typedef struct __declspec(align(2)) {
    unsigned short x;
} float16;

typedef struct __declspec(align(4)) {
    unsigned int x;
} float32;

#else

typedef struct {
    unsigned short x;
} __attribute__((aligned(2))) float16;

typedef struct {
    unsigned int x;
} __attribute__((aligned(4))) float32;

#endif

inline const TypeMeta& TypeStringToMeta(const std::string& str_type) {
    static std::unordered_map<std::string, TypeMeta> s2m_type_map {
        { "float32", TypeMeta::Make<float>() },
        { "int32", TypeMeta::Make<int>() },
        { "int64", TypeMeta::Make<int64_t>() },
        { "float64", TypeMeta::Make<double>() },
        { "float16", TypeMeta::Make<float16>() },
        { "uint8", TypeMeta::Make<uint8_t>() } 
    };
    static TypeMeta unknown_type;
    return s2m_type_map.count(str_type) ? s2m_type_map[str_type] : unknown_type;
}

inline const std::string TypeMetaToString(const TypeMeta& meta) {
    static std::unordered_map<TypeId, std::string> m2s_type_map {
        { TypeMeta::Id<float>(), "float32" },
        { TypeMeta::Id<int>(), "int32" },
        { TypeMeta::Id<int64_t>(), "int64" },
        { TypeMeta::Id<double>(), "float64", },
        { TypeMeta::Id<float16>(), "float16" },
        { TypeMeta::Id<uint8_t>(), "uint8" } 
    };
    return m2s_type_map.count(meta.id()) ? m2s_type_map[meta.id()] : "unknown";
}

}    // namespace dragon

#endif  // DRAGON_CORE_TYPES_H_