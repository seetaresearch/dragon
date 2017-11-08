// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_CORE_COMMON_H_
#define DRAGON_CORE_COMMON_H_

#include <climits>
#include <memory>
#include <string>
#include <queue>
#include <stack>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <mutex>

#include "core/types.h"
#include "protos/dragon.pb.h"
#include "utils/logging.h"

namespace dragon {

using std::string;
using std::queue;
using std::stack;
using std::vector;
using std::pair;
using std::set;
using std::map;
using std::mutex;
using std::unique_ptr;
using std::shared_ptr;

template <typename Key, typename Value>
using Map = std::unordered_map<Key, Value>;

template <typename Value>
using Set = std::unordered_set<Value> ;

#define CONCATENATE_IMPL(s1, s2) s1##s2
#define CONCATENATE(s1, s2) CONCATENATE_IMPL(s1,s2)
#define ANONYMOUS_VARIABLE(str) CONCATENATE(str, __LINE__)
#define NOT_IMPLEMENTED LOG(FATAL) << "This module has not been implemented yet."

}    // namespace dragon

#endif    // DRAGON_CORE_COMMON_H_