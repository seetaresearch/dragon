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
template <class KeyT, class ClassT, class... Args>
class Registry {
 public:
  typedef std::function<ClassT*(Args...)> Creator;

  /*! \brief Constructor with the name */
  explicit Registry(const string& name) : name_(name) {}

  /*! \brief Create an instance of specified class */
  ClassT* Create(const KeyT& key, Args... args) {
    CHECK(registry_.count(key))
        << "\n'" << key << "' is not registered in " << name_ << ".";
    return registry_[key](args...);
  }

  /*! \brief Return whether the specified class is registered */
  bool Has(const KeyT& key) {
    return (registry_.count(key)) != 0;
  }

  /*! \brief Register a class with the creator */
  void Register(const KeyT& key, Creator creator) {
    CHECK(!registry_.count(key))
        << "\n'" << key << "' has already registered in " << name_ << ".";
    registry_[key] = creator;
  }

  /*! \brief Return the key of registered classes */
  vector<KeyT> keys() {
    vector<KeyT> ret;
    for (const auto& it : registry_) {
      ret.push_back(it.first);
    }
    return ret;
  }

 private:
  /*! \brief The registry name */
  std::string name_;

  /*! \brief The registry map */
  Map<KeyT, Creator> registry_;
};

/*!
 * \brief Register creator into the registry.
 */
template <class KeyT, class ClassT, class... Args>
class Registerer {
 public:
  /*! \brief Constructor with key and creator */
  Registerer(
      const KeyT& key,
      Registry<KeyT, ClassT, Args...>* registry,
      typename Registry<KeyT, ClassT, Args...>::Creator creator,
      const string& help_msg = "") {
    registry->Register(key, creator);
  }

  /*! \brief Return the default creator */
  template <class DerivedT>
  static ClassT* DefaultCreator(Args... args) {
    return new DerivedT(args...);
  }
};

// Used in *.h files.
#define DECLARE_TYPED_REGISTRY(RegistryName, KeyT, ClassT, ...)     \
  DRAGON_API Registry<KeyT, ClassT, ##__VA_ARGS__>* RegistryName(); \
  typedef Registerer<KeyT, ClassT, ##__VA_ARGS__> Registerer##RegistryName;

// Used in *.cc files.
#define DEFINE_TYPED_REGISTRY(RegistryName, KeyT, ClassT, ...)    \
  Registry<KeyT, ClassT, ##__VA_ARGS__>* RegistryName() {         \
    static Registry<KeyT, ClassT, ##__VA_ARGS__>* registry =      \
        new Registry<KeyT, ClassT, ##__VA_ARGS__>(#RegistryName); \
    return registry;                                              \
  }

#define DECLARE_REGISTRY(RegistryName, ClassT, ...) \
  DECLARE_TYPED_REGISTRY(RegistryName, string, ClassT, ##__VA_ARGS__)

#define DEFINE_REGISTRY(RegistryName, ClassT, ...) \
  DEFINE_TYPED_REGISTRY(RegistryName, string, ClassT, ##__VA_ARGS__)

#define REGISTER_TYPED_CLASS(RegistryName, key, ...)                    \
  static Registerer##RegistryName ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key,                                                              \
      RegistryName(),                                                   \
      Registerer##RegistryName::DefaultCreator<__VA_ARGS__>)

#define REGISTER_CLASS(RegistryName, key, ...) \
  REGISTER_TYPED_CLASS(RegistryName, #key, __VA_ARGS__)

} // namespace dragon

#endif // DRAGON_CORE_REGISTRY_H_
