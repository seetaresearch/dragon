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

#ifndef DRAGON_CORE_COMMON_H_
#define DRAGON_CORE_COMMON_H_

#include <float.h>
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstring>
#include <ctime>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dragon/core/types.h"
#include "dragon/proto/dragon.pb.h"
#include "dragon/utils/logging.h"
#include "dragon/utils/macros.h"
#include "dragon/utils/string_utils.h"

namespace dragon {

using std::map;
using std::pair;
using std::queue;
using std::set;
using std::shared_ptr;
using std::stack;
using std::string;
using std::unique_ptr;
using std::vector;

template <typename Key, typename Value>
using Map = std::unordered_map<Key, Value>;

template <typename Value>
using Set = std::unordered_set<Value>;

// Fix the random seed for reproducing
#define DEFAULT_RNG_SEED 3

} // namespace dragon

#endif // DRAGON_CORE_COMMON_H_
