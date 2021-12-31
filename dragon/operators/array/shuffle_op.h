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

#ifndef DRAGON_OPERATORS_ARRAY_SHUFFLE_OP_H_
#define DRAGON_OPERATORS_ARRAY_SHUFFLE_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class ChannelShuffleOp final : public Operator<Context> {
 public:
  ChannelShuffleOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        group_(OP_SINGLE_ARG(int64_t, "group", 1)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t group_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_SHUFFLE_OP_H_
