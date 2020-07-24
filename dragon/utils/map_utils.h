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

#ifndef DRAGON_UTILS_MAP_UTILS_H_
#define DRAGON_UTILS_MAP_UTILS_H_

#include <utility>

namespace dragon {

template <
    class Map,
    typename Key = typename Map::key_type,
    typename Value = typename Map::mapped_type>
typename Map::mapped_type
get_default(const Map& map, const Key& key, Value&& default_value) {
  using M = typename Map::mapped_type;
  auto pos = map.find(key);
  return pos != map.end() ? (pos->second)
                          : M(std::forward<Value>(default_value));
}

} // namespace dragon

#endif // DRAGON_UTILS_MAP_UTILS_H_
