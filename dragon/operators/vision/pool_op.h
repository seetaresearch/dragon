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

#ifndef DRAGON_OPERATORS_VISION_POOL_OP_H_
#define DRAGON_OPERATORS_VISION_POOL_OP_H_

#include "dragon/operators/vision/pool_op_base.h"

namespace dragon {

template <class Context>
class PoolOp final : public PoolOpBase<Context> {
 public:
  explicit PoolOp(const OperatorDef& def, Workspace* ws)
      : PoolOpBase<Context>(def, ws) {
    GetBaseArguments();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_POOL_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class PoolGradientOp final : public PoolOpBase<Context> {
 public:
  PoolGradientOp(const OperatorDef& def, Workspace* ws)
      : PoolOpBase<Context>(def, ws) {
    GetBaseArguments();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_POOL_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_CUDNN
template <class Context>
class CuDNNPoolOp final : public CuDNNPoolOpBase<Context> {
 public:
  CuDNNPoolOp(const OperatorDef& def, Workspace* ws)
      : CuDNNPoolOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_POOL_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class CuDNNPoolGradientOp final : public CuDNNPoolOpBase<Context> {
 public:
  CuDNNPoolGradientOp(const OperatorDef& def, Workspace* ws)
      : CuDNNPoolOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_POOL_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};
#endif // USE_CUDNN

#ifdef USE_MPS
template <class Context>
class MPSPoolOp final : public MPSPoolOpBase<Context> {
 public:
  MPSPoolOp(const OperatorDef& def, Workspace* ws)
      : MPSPoolOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_POOL_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class MPSPoolGradientOp final : public MPSPoolOpBase<Context> {
 public:
  MPSPoolGradientOp(const OperatorDef& def, Workspace* ws)
      : MPSPoolOpBase<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_POOL_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};
#endif // USE_MPS

#ifdef USE_MLU
template <class Context>
class CNNLPoolOp final : public CNNLPoolOpBase<Context> {
 public:
  CNNLPoolOp(const OperatorDef& def, Workspace* ws)
      : CNNLPoolOpBase<Context>(def, ws) {
    CHECK_EQ(data_format(), "NHWC");
  }
  USE_OPERATOR_FUNCTIONS;
  USE_POOL_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class CNNLPoolGradientOp final : public CNNLPoolOpBase<Context> {
 public:
  CNNLPoolGradientOp(const OperatorDef& def, Workspace* ws)
      : CNNLPoolOpBase<Context>(def, ws) {
    CHECK_EQ(data_format(), "NHWC");
  }
  USE_OPERATOR_FUNCTIONS;
  USE_POOL_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_POOL_OP_H_
