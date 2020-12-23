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

#ifndef DRAGON_OPERATORS_METRIC_ACCURACY_OP_H_
#define DRAGON_OPERATORS_METRIC_ACCURACY_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class AccuracyOp final : public Operator<Context> {
 public:
  AccuracyOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        top_k_(OP_SINGLE_ARG(int64_t, "top_k", 1)),
        ignore_index_(OP_SINGLE_ARG(int64_t, "ignore_index", -1)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename LogitT, typename TargetT>
  void DoRunWithType();

 protected:
  int64_t top_k_, ignore_index_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_METRIC_ACCURACY_OP_H_
