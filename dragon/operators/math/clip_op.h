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

#ifndef DRAGON_OPERATORS_MATH_CLIP_OP_H_
#define DRAGON_OPERATORS_MATH_CLIP_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class ClipOp : public Operator<Context> {
 public:
  ClipOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        low_(OP_SINGLE_ARG(float, "low", -FLT_MAX)),
        high_(OP_SINGLE_ARG(float, "high", FLT_MAX)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
  }

  template <typename T>
  std::pair<float, float> GetLimits() {
    float type_min, type_max;
    const auto meta = TypeMeta::Make<T>();
    if (meta.template Match<uint8_t>()) {
      type_min = float(std::numeric_limits<uint8_t>::min());
      type_max = float(std::numeric_limits<uint8_t>::max());
    } else if (meta.template Match<int8_t>()) {
      type_min = float(std::numeric_limits<int8_t>::min());
      type_max = float(std::numeric_limits<int8_t>::max());
    } else if (meta.template Match<int>()) {
      type_min = float(std::numeric_limits<int>::min());
      type_max = float(std::numeric_limits<int>::max());
    } else if (meta.template Match<float16>()) {
      type_min = -65505.f, type_max = 65504.f;
    } else {
      type_min = std::numeric_limits<float>::min();
      type_max = std::numeric_limits<float>::max();
    }
    return std::make_pair(std::max(low_, type_min), std::min(high_, type_max));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float low_, high_;
};

template <class Context>
class ClipGradientOp final : public ClipOp<Context> {
 public:
  ClipGradientOp(const OperatorDef& def, Workspace* ws)
      : ClipOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

} // namespace dragon

#endif // DRAGON_OPERATORS_MATH_CLIP_OP_H_
