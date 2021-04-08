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

#ifndef DRAGON_MODULES_PYTHON_TYPES_H_
#define DRAGON_MODULES_PYTHON_TYPES_H_

#include <numpy/arrayobject.h>

#include "dragon/core/types.h"
#include "dragon/utils/dlpack.h"

namespace dragon {

namespace python {

namespace dtypes {

inline const int to_npy(const TypeMeta& type) {
  static std::unordered_map<TypeId, int> m{
      {TypeMeta::Id<bool>(), NPY_BOOL},
      {TypeMeta::Id<uint8_t>(), NPY_UINT8},
      {TypeMeta::Id<int8_t>(), NPY_INT8},
      {TypeMeta::Id<int>(), NPY_INT32},
      {TypeMeta::Id<int64_t>(), NPY_INT64},
      {TypeMeta::Id<float16>(), NPY_FLOAT16},
      {TypeMeta::Id<float>(), NPY_FLOAT32},
      {TypeMeta::Id<double>(), NPY_FLOAT64},
      {TypeMeta::Id<std::string>(), NPY_OBJECT},
  };
  auto it = m.find(type.id());
  return it != m.end() ? it->second : -1;
}

inline const TypeMeta& from_npy(int type) {
  static TypeMeta unknown_type;
  static std::unordered_map<int, TypeMeta> m{
      {NPY_BOOL, TypeMeta::Make<bool>()},
      {NPY_UINT8, TypeMeta::Make<uint8_t>()},
      {NPY_INT8, TypeMeta::Make<int8_t>()},
      {NPY_INT32, TypeMeta::Make<int>()},
      {NPY_INT64, TypeMeta::Make<int64_t>()},
      {NPY_FLOAT16, TypeMeta::Make<float16>()},
      {NPY_FLOAT32, TypeMeta::Make<float>()},
      {NPY_FLOAT64, TypeMeta::Make<double>()},
      {NPY_UNICODE, TypeMeta::Make<std::string>()},
      {NPY_STRING, TypeMeta::Make<std::string>()},
  };
  auto it = m.find(type);
  return it != m.end() ? it->second : unknown_type;
}

inline DLDataType* to_dlpack(const TypeMeta& type) {
  static std::unordered_map<TypeId, DLDataType> m{
      {TypeMeta::Id<bool>(), DLDataType{1, 8, 1}},
      {TypeMeta::Id<uint8_t>(), DLDataType{1, 8, 1}},
      {TypeMeta::Id<int8_t>(), DLDataType{0, 8, 1}},
      {TypeMeta::Id<int>(), DLDataType{0, 32, 1}},
      {TypeMeta::Id<int64_t>(), DLDataType{0, 64, 1}},
      {TypeMeta::Id<float16>(), DLDataType{2, 16, 1}},
      {TypeMeta::Id<float>(), DLDataType{2, 32, 1}},
      {TypeMeta::Id<double>(), DLDataType{2, 64, 1}},
  };
  auto it = m.find(type.id());
  return it != m.end() ? &(it->second) : nullptr;
}

inline const TypeMeta& from_dlpack(const DLDataType& type) {
  static TypeMeta unknown_type;
  static std::unordered_map<int, std::unordered_map<int, TypeMeta>> m{
      {0,
       std::unordered_map<int, TypeMeta>{
           {8, TypeMeta::Make<int8_t>()},
           {32, TypeMeta::Make<int32_t>()},
           {64, TypeMeta::Make<int64_t>()},
       }},
      {1,
       std::unordered_map<int, TypeMeta>{
           {8, TypeMeta::Make<uint8_t>()},
       }},
      {2,
       std::unordered_map<int, TypeMeta>{
           {16, TypeMeta::Make<float16>()},
           {32, TypeMeta::Make<float>()},
           {64, TypeMeta::Make<double>()},
       }},
  };
  if (type.lanes != 1) return unknown_type;
  if (!m.count(type.code)) return unknown_type;
  const auto& mm = m.at(type.code);
  if (!mm.count(type.bits)) return unknown_type;
  return mm.at(type.bits);
}

} // namespace dtypes

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_TYPES_H_
