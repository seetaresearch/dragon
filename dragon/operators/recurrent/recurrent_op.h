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

#ifndef DRAGON_OPERATORS_RECURRENT_CUDNN_RECURRENT_OP_H_
#define DRAGON_OPERATORS_RECURRENT_CUDNN_RECURRENT_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class RecurrentOp : public Operator<Context> {
 public:
  RecurrentOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    LOG(FATAL) << "CuDNN is required.";
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {}
};

template <class Context>
class RecurrentGradientOp : public Operator<Context> {
 public:
  RecurrentGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    LOG(FATAL) << "CuDNN is required.";
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {}
};

} // namespace dragon

#endif // DRAGON_OPERATORS_RECURRENT_RECURRENT_OP_H_
