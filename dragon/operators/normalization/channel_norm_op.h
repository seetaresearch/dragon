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

#ifndef DRAGON_OPERATORS_NORMALIZATION_CHANNEL_NORM_OP_H_
#define DRAGON_OPERATORS_NORMALIZATION_CHANNEL_NORM_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class ChannelNormOp final : public Operator<Context> {
 public:
  ChannelNormOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    INITIALIZE_OP_REPEATED_ARG(int64_t, perm);
    auto mean = OP_REPEATED_ARG(float, "mean");
    auto std = OP_REPEATED_ARG(float, "std");
    CHECK_EQ(mean.size(), std.size())
        << "\nSize of <mean> and <std> should be same.";
    X_mean_.Reshape({(int64_t)mean.size()});
    X_std_.Reshape({(int64_t)std.size()});
    auto* m = X_mean_.template mutable_data<float, CPUContext>();
    auto* s = X_std_.template mutable_data<float, CPUContext>();
    for (size_t i = 0; i < mean.size(); ++i) {
      m[i] = mean[i], s[i] = std[i];
    }
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename Tx, typename Ty>
  void DoRunWithTypeAndCast();

  template <typename T>
  void DoRunWithType();

 protected:
  Tensor X_mean_, X_std_;
  DECLARE_OP_REPEATED_ARG(int64_t, perm);
};

DEFINE_OP_REPEATED_ARG(int64_t, ChannelNormOp, perm);

} // namespace dragon

#endif // DRAGON_OPERATORS_NORMALIZATION_CHANNEL_NORM_OP_H_
