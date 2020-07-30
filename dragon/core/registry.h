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

#ifndef DRAGON_CORE_REGISTRY_H_
#define DRAGON_CORE_REGISTRY_H_

#include "dragon/core/common.h"

namespace dragon {

/*!
 * \brief Registry to create class instances.
 */
template <class KeyType, class ObjectType, class... Args>
class Registry {
 public:
  typedef std::function<ObjectType*(Args...)> Creator;

  /*! \brief Create an instance of specified class */
  ObjectType* Create(const KeyType& key, Args... args) {
    CHECK(registry_.count(key)) << "\nKey(" << key << ") has not registered.";
    return registry_[key](args...);
  }

  /*! \brief Return whether the specified class is registered */
  bool Has(const KeyType& key) {
    return (registry_.count(key)) != 0;
  }

  /*! \brief Register a class with the creator */
  void Register(const KeyType& key, Creator creator) {
    CHECK(!registry_.count(key))
        << "\nKey(" << key << ") has already registered.";
    registry_[key] = creator;
  }

  /*! \brief Return the key of registered classes */
  vector<KeyType> keys() {
    vector<KeyType> ret;
    for (const auto& it : registry_) {
      ret.push_back(it.first);
    }
    return ret;
  }

 private:
  /*! \brief The registry map */
  Map<KeyType, Creator> registry_;
};

/*!
 * \brief Register creator into the registry.
 */
template <class KeyType, class ObjectType, class... Args>
class Registerer {
 public:
  /*! \brief Constructor with key and creator */
  Registerer(
      const KeyType& key,
      Registry<KeyType, ObjectType, Args...>* registry,
      typename Registry<KeyType, ObjectType, Args...>::Creator creator,
      const string& help_msg = "") {
    registry->Register(key, creator);
  }

  /*! \brief Return the default creator */
  template <class DerivedType>
  static ObjectType* DefaultCreator(Args... args) {
    return new DerivedType(args...);
  }
};

// Used in *.h files
#define DECLARE_TYPED_REGISTRY(RegistryName, KeyType, ObjectType, ...)     \
  DRAGON_API Registry<KeyType, ObjectType, ##__VA_ARGS__>* RegistryName(); \
  typedef Registerer<KeyType, ObjectType, ##__VA_ARGS__>                   \
      Registerer##RegistryName;

// Used in *.cc files
#define DEFINE_TYPED_REGISTRY(RegistryName, KeyType, ObjectType, ...) \
  Registry<KeyType, ObjectType, ##__VA_ARGS__>* RegistryName() {      \
    static Registry<KeyType, ObjectType, ##__VA_ARGS__>* registry =   \
        new Registry<KeyType, ObjectType, ##__VA_ARGS__>();           \
    return registry;                                                  \
  }

#define DECLARE_REGISTRY(RegistryName, ObjectType, ...) \
  DECLARE_TYPED_REGISTRY(RegistryName, string, ObjectType, ##__VA_ARGS__)

#define DEFINE_REGISTRY(RegistryName, ObjectType, ...) \
  DEFINE_TYPED_REGISTRY(RegistryName, string, ObjectType, ##__VA_ARGS__)

#define REGISTER_TYPED_CLASS(RegistryName, key, ...)                    \
  static Registerer##RegistryName ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key,                                                              \
      RegistryName(),                                                   \
      Registerer##RegistryName::DefaultCreator<__VA_ARGS__>)

#define REGISTER_CLASS(RegistryName, key, ...) \
  REGISTER_TYPED_CLASS(RegistryName, #key, __VA_ARGS__)

} // namespace dragon

#endif // DRAGON_CORE_REGISTRY_H_
