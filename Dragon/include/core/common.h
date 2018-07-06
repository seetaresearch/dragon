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

#ifndef DRAGON_CORE_COMMON_H_
#define DRAGON_CORE_COMMON_H_

#include <ctime>
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
#include <functional>

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
using std::unique_ptr;
using std::shared_ptr;

template <typename Key, typename Value>
using Map = std::unordered_map<Key, Value>;

template <typename Value>
using Set = std::unordered_set<Value> ;

/*
 * Define the Kernel version.
 *
 * | Major(2) | Minor(2) | Patch(07) |
 */
#define DRAGON_VERSION 2207

/*
 * Define the default random seed.
 */
#define DEFAULT_RNG_SEED 3

/*
 * Define the common marcos.
 */
#ifdef _MSC_VER
#if _MSC_VER < 1900
#define thread_local __declspec(thread)
#endif
#endif

#define CONCATENATE_IMPL(s1, s2) s1##s2
#define CONCATENATE(s1, s2) CONCATENATE_IMPL(s1,s2)
#define ANONYMOUS_VARIABLE(str) CONCATENATE(str, __LINE__)
#define NOT_IMPLEMENTED LOG(FATAL) << "This module has not been implemented yet."

}    // namespace dragon

#endif    // DRAGON_CORE_COMMON_H_