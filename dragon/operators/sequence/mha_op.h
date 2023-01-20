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

#ifndef DRAGON_OPERATORS_SEQUENCE_MHA_OP_H_
#define DRAGON_OPERATORS_SEQUENCE_MHA_OP_H_

#include "dragon/operators/sequence/mha_op_base.h"

namespace dragon {

template <class Context>
class MultiHeadSelfAttnOp : public Operator<Context> {
 public:
  MultiHeadSelfAttnOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    NOT_IMPLEMENTED;
  }
};

#ifdef USE_CUDNN
template <class Context>
class CuDNNMultiHeadSelfAttnOp : public CuDNNMultiHeadSelfAttnOpBase<Context> {
 public:
  CuDNNMultiHeadSelfAttnOp(const OperatorDef& def, Workspace* ws)
      : CuDNNMultiHeadSelfAttnOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class CuDNNMultiHeadSelfAttnGradientOp
    : public CuDNNMultiHeadSelfAttnOpBase<Context> {
 public:
  CuDNNMultiHeadSelfAttnGradientOp(const OperatorDef& def, Workspace* ws)
      : CuDNNMultiHeadSelfAttnOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};
#endif // USE_CUDNN

#ifdef USE_MLU
template <class Context>
class CNNLMultiHeadSelfAttnOp : public CNNLMultiHeadSelfAttnOpBase<Context> {
 public:
  CNNLMultiHeadSelfAttnOp(const OperatorDef& def, Workspace* ws)
      : CNNLMultiHeadSelfAttnOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class CNNLMultiHeadSelfAttnGradientOp
    : public CNNLMultiHeadSelfAttnOpBase<Context> {
 public:
  CNNLMultiHeadSelfAttnGradientOp(const OperatorDef& def, Workspace* ws)
      : CNNLMultiHeadSelfAttnOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_SEQUENCE_MHA_OP_H_
