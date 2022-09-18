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

/*!
 * \brief The meta class for all types.
 *
 * TypeMeta is commonly used for type identification:
 *
 * \code{.cpp}
 * auto meta1 = dragon::TypeMeta::Make<float>();
 * auto meta2 = dragon::TypeMeta::Make<float>();
 * std::cout << (meta1 == meta2) << std::endl; // 1
 * std::cout << (meta1.id() == meta2.id()) << std::endl; // 1
 * std::cout << meta1.Match<float>() << std::endl; // 1
 * std::cout << (meta1.id() == dragon::TypeMeta::Id<float>()) << std::endl; // 1
 * \endcode
 *
 * Default constructor and destructor are available for non-fundamental types:
 *
 * \code{.cpp}
 * auto meta = dragon::TypeMeta::Make<std::string>();
 * auto* raw_string_data = malloc(1 * meta.itemsize());
 * meta.ctor()(raw_string_data, 1);
 * auto* string_data = reinterpret_cast<std::string*>(raw_string_data);
 * std::cout << string_data[0].size();
 * meta.dtor()(raw_string_data, 1);
 * \endcode
 */
class TypeMeta {
 public:
  typedef void (*PlacementNew)(void*, size_t);
  typedef void (*TypedCopy)(const void*, void*, size_t);
  typedef void (*TypedDestructor)(void*, size_t);

  /*! \brief Default constructor */
  TypeMeta() : id_(0), itemsize_(0) {}

  /*! \brief Constructor with the other type meta */
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

  /*! \brief Return whether the two identifications are equal */
  bool operator==(const TypeMeta& other) const {
    return (id_ == other.id_);
  }

  /*! \brief Return whether the two identifications are not equal */
  bool operator!=(const TypeMeta& other) const {
    return (id_ != other.id_);
  }

  /*! \brief Return the identification of given type */
  template <typename T>
  static TypeId Id() {
    return TypeRegister<T>::id();
  }

  /*! \brief Return the item size of given type */
  template <typename T>
  static size_t Itemsize() {
    return sizeof(T);
  }

  /*! \brief Call the constructor for each element */
  template <typename T>
  static void Ctor(void* ptr, size_t n) {
    T* typed_ptr = static_cast<T*>(ptr);
    for (size_t i = 0; i < n; ++i) {
      new (typed_ptr + i) T;
    }
  }

  /*! \brief Call the destructor for each element */
  template <typename T>
  static void Dtor(void* ptr, size_t n) {
    T* typed_ptr = static_cast<T*>(ptr);
    for (size_t i = 0; i < n; ++i) {
      typed_ptr[i].~T();
    }
  }

  /*! \brief Call the copy constructor for each element */
  template <typename T>
  static void Copy(const void* src, void* dst, size_t n) {
    const T* typed_src = static_cast<const T*>(src);
    T* typed_dst = static_cast<T*>(dst);
    for (size_t i = 0; i < n; ++i) {
      typed_dst[i] = typed_src[i];
    }
  }

#define FundamentalTypeMeta \
  std::enable_if<std::is_fundamental<T>::value, TypeMeta>::type

#define StructuralTypeMeta                                                 \
  std::enable_if<                                                          \
      !std::is_fundamental<T>::value && std::is_copy_assignable<T>::value, \
      TypeMeta>::type

  /*! \brief Return a type meta of given type */
  template <typename T>
  static typename FundamentalTypeMeta Make() {
    return TypeMeta(Id<T>(), Itemsize<T>(), nullptr, nullptr, nullptr);
  }

  /*! \brief Return a type meta of given type */
  template <typename T>
  static typename StructuralTypeMeta Make() {
    return TypeMeta(Id<T>(), Itemsize<T>(), Ctor<T>, Copy<T>, Dtor<T>);
  }

#undef FundamentalTypeMeta
#undef StructuralTypeMeta

  /*! \brief Return whether the meta is matched with given type */
  template <typename T>
  bool Match() const {
    return (id_ == Id<T>());
  }

  /*! \brief Return the type identification */
  const TypeId& id() const {
    return id_;
  }

  /*! \brief Return the item size */
  const size_t& itemsize() const {
    return itemsize_;
  }

  /*! \brief Return the type constructor */
  PlacementNew ctor() const {
    return ctor_;
  }

  /*! \brief Return the type destructor */
  TypedDestructor dtor() const {
    return dtor_;
  }

  /*! \brief Return the type copy constructor */
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
  PlacementNew ctor_ = nullptr;
  TypedCopy copy_ = nullptr;
  TypedDestructor dtor_ = nullptr;
};

} // namespace dragon

#endif // DRAGON_CORE_TYPEID_H_
