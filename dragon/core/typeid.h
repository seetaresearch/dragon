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

#ifndef DRAGON_CORE_TYPEID_H_
#define DRAGON_CORE_TYPEID_H_

#include <cstdlib>
#include <iostream>
#include <map>

#include "dragon/utils/macros.h"

namespace dragon {

typedef intptr_t TypeId;

template <typename T>
struct DRAGON_API TypeRegister {
  static TypeId id() {
    static bool type_id_bit[1];
    return (TypeId)(type_id_bit);
  }
};

class TypeMeta {
 public:
  typedef void (*PlacementNew)(void*, size_t);
  typedef void (*TypedCopy)(const void*, void*, size_t);
  typedef void (*TypedDestructor)(void*, size_t);

  TypeMeta()
      : id_(0), itemsize_(0), ctor_(nullptr), copy_(nullptr), dtor_(nullptr) {}

  TypeMeta(const TypeMeta& src)
      : id_(src.id_),
        itemsize_(src.itemsize_),
        ctor_(src.ctor_),
        copy_(src.copy_),
        dtor_(src.dtor_) {}

  TypeMeta& operator=(const TypeMeta& src) {
    if (this == &src) return *this;
    id_ = src.id_;
    itemsize_ = src.itemsize_;
    ctor_ = src.ctor_;
    copy_ = src.copy_;
    dtor_ = src.dtor_;
    return *this;
  }

  bool operator==(const TypeMeta& other) const {
    return (id_ == other.id_);
  }

  bool operator!=(const TypeMeta& other) const {
    return (id_ != other.id_);
  }

  template <typename T>
  static TypeId Id() {
    return TypeRegister<T>::id();
  }

  template <typename T>
  static size_t Itemsize() {
    return sizeof(T);
  }

  template <typename T>
  static void Ctor(void* ptr, size_t n) {
    T* typed_ptr = static_cast<T*>(ptr);
    for (size_t i = 0; i < n; i++) {
      new (typed_ptr + i) T;
    }
  }

  template <typename T>
  static void Dtor(void* ptr, size_t n) {
    T* typed_ptr = static_cast<T*>(ptr);
    for (size_t i = 0; i < n; ++i) {
      typed_ptr[i].~T();
    }
  }

  template <typename T>
  static void Copy(const void* src, void* dst, size_t n) {
    const T* typed_src = static_cast<const T*>(src);
    T* typed_dst = static_cast<T*>(dst);
    for (size_t i = 0; i < n; ++i)
      typed_dst[i] = typed_src[i];
  }

#define FundMeta std::enable_if<std::is_fundamental<T>::value, TypeMeta>::type

#define StructMeta                                                         \
  std::enable_if<                                                          \
      !std::is_fundamental<T>::value && std::is_copy_assignable<T>::value, \
      TypeMeta>::type

  template <typename T>
  static typename FundMeta Make() {
    return TypeMeta(Id<T>(), Itemsize<T>(), nullptr, nullptr, nullptr);
  }

  template <typename T>
  static typename StructMeta Make() {
    return TypeMeta(Id<T>(), Itemsize<T>(), Ctor<T>, Copy<T>, Dtor<T>);
  }

#undef FundMeta
#undef StructMeta

  template <typename T>
  bool Match() const {
    return (id_ == Id<T>());
  }

  const TypeId& id() const {
    return id_;
  }

  const size_t& itemsize() const {
    return itemsize_;
  }

  PlacementNew ctor() const {
    return ctor_;
  }

  TypedDestructor dtor() const {
    return dtor_;
  }

  TypedCopy copy() const {
    return copy_;
  }

 private:
  TypeMeta(
      TypeId id,
      size_t itemsize,
      PlacementNew ctor,
      TypedCopy copy,
      TypedDestructor dtor)
      : id_(id), itemsize_(itemsize), ctor_(ctor), copy_(copy), dtor_(dtor) {}

 private:
  TypeId id_;
  size_t itemsize_;
  PlacementNew ctor_;
  TypedCopy copy_;
  TypedDestructor dtor_;
};

} // namespace dragon

#endif // DRAGON_CORE_TYPEID_H_
