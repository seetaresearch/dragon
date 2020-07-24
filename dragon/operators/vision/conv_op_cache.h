/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * Codes are based on:
 *
 *     <https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op_cache_cudnn.h>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_VISION_CONV_OP_CACHE_H_
#define DRAGON_OPERATORS_VISION_CONV_OP_CACHE_H_

#include "dragon/core/common.h"

namespace dragon {

template <typename Algorithm>
class ConvAlgorithmCache {
 public:
  Algorithm get(
      const vec64_t& Xdims,
      const vec64_t& Wdims,
      int flag,
      std::function<Algorithm()> gen_func) {
    int64_t key = 0;
    std::hash<int64_t> hash_func;

    for (auto dim : Xdims) {
      key ^= hash_func(dim) + 0x9e3779b9 + (key << 6) + (key >> 2);
    }

    for (auto dim : Wdims) {
      key ^= hash_func(dim) + 0x9e3779b9 + (key << 6) + (key >> 2) + 1;
    }

    key ^= hash_func(flag) + 0x9e3779b9 + (key << 6) + (key >> 2) + 2;

    if (map_.find(key) == map_.end()) {
      auto value = gen_func();
      map_[key] = value;
    }

    return map_[key];
  }

 private:
  Map<int64_t, Algorithm> map_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_CONV_OP_CACHE_H_
