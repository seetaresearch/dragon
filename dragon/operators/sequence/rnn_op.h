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

#ifndef DRAGON_OPERATORS_SEQUENCE_RNN_OP_H_
#define DRAGON_OPERATORS_SEQUENCE_RNN_OP_H_

#include "dragon/operators/sequence/rnn_op_base.h"

namespace dragon {

#ifdef USE_CUDNN
template <class Context>
class CuDNNRNNOp final : public CuDNNRNNOpBase<Context> {
 public:
  CuDNNRNNOp(const OperatorDef& def, Workspace* ws)
      : CuDNNRNNOpBase<Context>(def, ws) {}
  USE_CUDNN_RNN_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class CuDNNRNNGradientOp final : public CuDNNRNNOpBase<Context> {
 public:
  CuDNNRNNGradientOp(const OperatorDef& def, Workspace* ws)
      : CuDNNRNNOpBase<Context>(def, ws) {}
  USE_CUDNN_RNN_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};
#endif // USE_CUDNN

} // namespace dragon

#endif // DRAGON_OPERATORS_SEQUENCE_RNN_OP_H_
