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

#ifndef DRAGON_OPERATORS_NORMALIZATION_LRN_OP_H_
#define DRAGON_OPERATORS_NORMALIZATION_LRN_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class LRNOp : public Operator<Context> {
 public:
  LRNOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        size_(OP_SINGLE_ARG(int64_t, "size", 5)),
        alpha_(OP_SINGLE_ARG(float, "alpha", 0.0001f)),
        beta_(OP_SINGLE_ARG(float, "beta", 0.75f)),
        bias_(OP_SINGLE_ARG(float, "bias", 1.f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t size_;
  float alpha_, beta_, bias_;
};

template <class Context>
class LRNGradientOp : public Operator<Context> {
 public:
  LRNGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        size_(OP_SINGLE_ARG(int64_t, "size", 5)),
        alpha_(OP_SINGLE_ARG(float, "alpha", 0.0001f)),
        beta_(OP_SINGLE_ARG(float, "beta", 0.75f)),
        bias_(OP_SINGLE_ARG(float, "bias", 1.f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t size_;
  float alpha_, beta_, bias_;
};

#ifdef USE_CUDNN

template <class Context>
class CuDNNLRNOp final : public LRNOp<Context> {
 public:
  CuDNNLRNOp(const OperatorDef& def, Workspace* ws) : LRNOp<Context>(def, ws) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc_));
    CUDNN_CHECK(cudnnCreateLRNDescriptor(&lrn_desc_));
    CUDNN_CHECK(cudnnSetLRNDescriptor(
        lrn_desc_, this->size_, this->alpha_, this->beta_, this->bias_));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CuDNNLRNOp() {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc_));
    CUDNN_CHECK(cudnnDestroyLRNDescriptor(lrn_desc_));
  }

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  cudnnTensorDescriptor_t input_desc_;
  cudnnLRNDescriptor_t lrn_desc_;
};

template <class Context>
class CuDNNLRNGradientOp final : public LRNGradientOp<Context> {
 public:
  CuDNNLRNGradientOp(const OperatorDef& def, Workspace* ws)
      : LRNGradientOp<Context>(def, ws) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc_));
    CUDNN_CHECK(cudnnCreateLRNDescriptor(&lrn_desc_));
    CUDNN_CHECK(cudnnSetLRNDescriptor(
        lrn_desc_, this->size_, this->alpha_, this->beta_, this->bias_));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CuDNNLRNGradientOp() {
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc_));
    CUDNN_CHECK(cudnnDestroyLRNDescriptor(lrn_desc_));
  }

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  cudnnTensorDescriptor_t input_desc_;
  cudnnLRNDescriptor_t lrn_desc_;
};

#endif // WITH CUDNN

} // namespace dragon

#endif // DRAGON_OPERATORS_NORMALIZATION_LRN_OP_H_
