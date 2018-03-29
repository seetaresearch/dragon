// ------------------------------------------------------------
// Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_CORE_REGISTRY_H_
#define DRAGON_CORE_REGISTRY_H_

#include <functional>

#include "core/common.h"
#include "utils/logging.h"

namespace dragon {

template <class SrcType, class ObjType, class... Args>
class Registry {
public:
    typedef std::function<ObjType*(Args ...)> Creator;
    void Register(const SrcType& key, Creator creator) {
        CHECK(!registry_.count(key)) << "Key(" << key  << ") has already registered.";
        registry_[key] = creator;
    }
    ObjType* Create(const SrcType& key, Args ... args) {
        CHECK(registry_.count(key)) << "Key(" << key << ") has not registered yet.";
        return registry_[key](args...);
    }
    bool Has(const SrcType& key) { return (registry_.count(key)) != 0; }
    vector<SrcType> keys() {
        vector<SrcType> ret;
        for (const auto& it : registry_) ret.push_back(it.first);
        return ret;
    }
private:
    Map<SrcType, Creator> registry_; 
};

template <class SrcType, class ObjType, class... Args>
class Registerer {
public:
    Registerer(const SrcType& key, Registry<SrcType, ObjType, Args...>* registry,
        typename Registry<SrcType, ObjType, Args...>::Creator creator, const string& help_msg = "") {
        registry->Register(key, creator);
    }
    template <class DerivedType>
    static ObjType* defaultCreator(Args ... args) {return new DerivedType(args...);}
};

//  use in *.h files
#define DECLARE_TYPED_REGISTRY(RegistryName, SrcType, ObjType,...) \
    dragon::Registry<SrcType, ObjType,##__VA_ARGS__>* RegistryName(); \
    typedef dragon::Registerer<SrcType,ObjType,##__VA_ARGS__> Registerer##RegistryName;

//  use in *.cc files
#define DEFINE_TYPED_REGISTRY(RegistryName,SrcType, ObjType,...) \
    Registry<SrcType,ObjType,##__VA_ARGS__>* RegistryName() {  \
        static Registry<SrcType,ObjType,##__VA_ARGS__>* registry = \
            new Registry<SrcType,ObjType,##__VA_ARGS__>(); \
        return registry; \
    }

#define DECLARE_REGISTRY(RegistryName, ObjType, ...) \
    DECLARE_TYPED_REGISTRY(RegistryName, string, ObjType, ##__VA_ARGS__)

#define DEFINE_REGISTRY(RegistryName, ObjType, ...) \
    DEFINE_TYPED_REGISTRY(RegistryName, string, ObjType, ##__VA_ARGS__)

#define REGISTER_TYPED_CLASS(RegistryName, key, ...) \
    static Registerer##RegistryName ANONYMOUS_VARIABLE(g_##RegistryName) ( \
    key, RegistryName(), Registerer##RegistryName::defaultCreator<__VA_ARGS__>)

#define REGISTER_CLASS(RegistryName, key, ...) \
    REGISTER_TYPED_CLASS(RegistryName, #key, __VA_ARGS__)

}    // namepsace dragon

#endif    //DRAGON_CORE_REGISTRY_H_
