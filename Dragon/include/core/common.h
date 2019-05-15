/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_CORE_COMMON_H_
#define DRAGON_CORE_COMMON_H_

#include <ctime>
#include <cmath>
#include <climits>
#include <float.h>
#include <random>
#include <numeric>
#include <memory>
#include <string>
#include <queue>
#include <stack>
#include <array>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <mutex>
#include <functional>

#include "core/types.h"
#include "proto/dragon.pb.h"
#include "utils/string.h"
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
using Set = std::unordered_set<Value>;

/* * * * * * * * * * * * * * * * * * * * *
 *                                       *
 *            Kernel Version             *
 *                                       *
 *     Major(3) | Minor(0) | Patch(00)   *
 *                                       *
 * * * * * * * * * * * * * * * * * * * * */

#define DRAGON_VERSION 3000

/* * * * * * * * * * * * * * * * * * * * *
 *                                       *
 *          Default Random Seed          *
 *                                       *
 * * * * * * * * * * * * * * * * * * * * */

#define DEFAULT_RNG_SEED 3

/* * * * * * * * * * * * * * * * * * * * *
 *                                       *
 *                Macros                 *
 *                                       *
 * * * * * * * * * * * * * * * * * * * * */

// Avoid using of "thread_local" for VS2013 or older Xcode
#if defined(__clang__) || defined(__GNUC__)
#define TLS_OBJECT __thread
#else
#define TLS_OBJECT __declspec(thread)
#endif

#define CONCATENATE_IMPL(s1, s2) s1##s2
#define CONCATENATE(s1, s2) CONCATENATE_IMPL(s1,s2)
#define ANONYMOUS_VARIABLE(str) CONCATENATE(str, __LINE__)
#define NOT_IMPLEMENTED LOG(FATAL) << "This module has not been implemented yet."

}  // namespace dragon

#endif  // DRAGON_CORE_COMMON_H_