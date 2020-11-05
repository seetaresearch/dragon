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

#ifndef DRAGON_OPERATORS_ACTIVATION_DROPOUT_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_DROPOUT_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class DropoutOp : public Operator<Context> {
 public:
  DropoutOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    INIT_OP_SINGLE_ARG_WITH_DESC(float, ratio, 0.5f);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_SINGLE_ARG_WITH_DESC(float, ratio);
};

template <class Context>
class DropoutGradientOp : public Operator<Context> {
 public:
  DropoutGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    INIT_OP_SINGLE_ARG_WITH_DESC(float, ratio, 0.5f);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_SINGLE_ARG_WITH_DESC(float, ratio);
};

DEFINE_OP_SINGLE_ARG_WITH_DESC(float, DropoutOp, ratio);
DEFINE_OP_SINGLE_ARG_WITH_DESC(float, DropoutGradientOp, ratio);

#ifdef USE_CUDNN

#if CUDNN_VERSION_MIN(7, 0, 0)

template <class Context>
class CuDNNDropoutOp final : public DropoutOp<Context> {
 public:
  CuDNNDropoutOp(const OperatorDef& def, Workspace* ws)
      : DropoutOp<Context>(def, ws),
        states_initialized_(false),
        rng_seed_(DEFAULT_RNG_SEED) {
    CuDNNCreateTensorDesc(&input_desc_);
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc_));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CuDNNDropoutOp() {
    CuDNNDestroyTensorDesc(&input_desc_);
    CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc_));
  }

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  bool states_initialized_;
  cudnnTensorDescriptor_t input_desc_;
  cudnnDropoutDescriptor_t dropout_desc_;
  unsigned long long rng_seed_;
};

template <class Context>
class CuDNNDropoutGradientOp final : public DropoutGradientOp<Context> {
 public:
  CuDNNDropoutGradientOp(const OperatorDef& def, Workspace* ws)
      : DropoutGradientOp<Context>(def, ws),
        states_initialized_(false),
        rng_seed_(DEFAULT_RNG_SEED) {
    CuDNNCreateTensorDesc(&input_desc_);
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc_));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CuDNNDropoutGradientOp() {
    CuDNNDestroyTensorDesc(&input_desc_);
    CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc_));
  }

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  bool states_initialized_;
  cudnnTensorDescriptor_t input_desc_;
  cudnnDropoutDescriptor_t dropout_desc_;
  unsigned long long rng_seed_;
};

#endif // CUDNN_VERSION_MIN(7, 0, 0)

#endif // USE_CUDNN

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_DROPOUT_OP_H_
