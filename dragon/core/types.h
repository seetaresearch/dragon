/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_CORE_TYPES_H_
#define DRAGON_CORE_TYPES_H_

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "dragon/core/typeid.h"

namespace dragon {

typedef std::vector<int> vec32_t;
typedef std::vector<int64_t> vec64_t;

#ifdef _MSC_VER

typedef struct __declspec(align(2)) {
  unsigned short x;
} float16;

#else

typedef struct {
  unsigned short x;
} __attribute__((aligned(2))) float16;

#endif

/*! \brief Array that packs a fixed number of elements */
template <typename T, int N>
struct SimpleArray {
  T data[N];
};

namespace types {

/*! \brief Convert the type string to meta */
inline const TypeMeta& to_meta(const std::string& type) {
  static TypeMeta unknown_type;
  static std::unordered_map<std::string, TypeMeta> m{
      {"bool", TypeMeta::Make<bool>()},
      {"int8", TypeMeta::Make<int8_t>()},
      {"uint8", TypeMeta::Make<uint8_t>()},
      {"int32", TypeMeta::Make<int>()},
      {"int64", TypeMeta::Make<int64_t>()},
      {"float16", TypeMeta::Make<float16>()},
      {"float32", TypeMeta::Make<float>()},
      {"float64", TypeMeta::Make<double>()},
      {"string", TypeMeta::Make<std::string>()},
  };
  auto it = m.find(type);
  return it != m.end() ? it->second : unknown_type;
}

/*! \brief Convert the type meta to string */
inline const std::string to_string(const TypeMeta& type) {
  static std::string unknown_type = "unknown";
  static std::unordered_map<TypeId, std::string> m{
      {TypeMeta::Id<bool>(), "bool"},
      {TypeMeta::Id<int8_t>(), "int8"},
      {TypeMeta::Id<uint8_t>(), "uint8"},
      {TypeMeta::Id<int>(), "int32"},
      {TypeMeta::Id<int64_t>(), "int64"},
      {TypeMeta::Id<float16>(), "float16"},
      {TypeMeta::Id<float>(), "float32"},
      {TypeMeta::Id<double>(), "float64"},
      {TypeMeta::Id<std::string>(), "string"},
  };
  auto it = m.find(type.id());
  return it != m.end() ? it->second : unknown_type;
}

/*! \brief Convert the type argument to string */
template <typename T>
inline const std::string to_string() {
  return to_string(TypeMeta::Make<T>());
}

} // namespace types

} // namespace dragon

#endif // DRAGON_CORE_TYPES_H_
