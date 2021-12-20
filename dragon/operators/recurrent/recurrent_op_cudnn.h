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

#ifndef DRAGON_OPERATORS_RECURRENT_RECURRENT_OP_CUDNN_H_
#define DRAGON_OPERATORS_RECURRENT_RECURRENT_OP_CUDNN_H_

#include "dragon/core/operator.h"

namespace dragon {

#ifdef USE_CUDNN

class CuDNNTensorDescs {
 public:
  CuDNNTensorDescs(int num_descs) {
    descs_.resize(num_descs);
    for (int i = 0; i < num_descs; ++i)
      CuDNNCreateTensorDesc(&descs_[i]);
  }

  ~CuDNNTensorDescs() {
    for (auto desc : descs_)
      CuDNNDestroyTensorDesc(&desc);
  }

  template <typename T>
  void Set(const vec64_t& dims, const vec64_t& strides) {
    CHECK_EQ(dims.size(), strides.size());
    for (auto desc : descs_)
      CuDNNSetTensorDesc<T>(&desc, dims, strides);
  }

  cudnnTensorDescriptor_t* data() {
    return descs_.data();
  }

 protected:
  vector<cudnnTensorDescriptor_t> descs_;
};

template <class Context>
class CuDNNRecurrentOpBase : public Operator<Context> {
 public:
  CuDNNRecurrentOpBase(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        states_initialized_(0),
        num_layers_(OP_SINGLE_ARG(int64_t, "num_layers", 1)),
        hidden_size_(OP_SINGLE_ARG(int64_t, "hidden_size", 0)),
        bidirectional_(OP_SINGLE_ARG(int64_t, "bidirectional", 0)),
        dropout_(OP_SINGLE_ARG(float, "dropout", 0.f)),
        rng_seed_(def.device_option().random_seed()),
        enable_tensor_core_(TENSOR_CORE_AVAILABLE() ? 1 : 0) {
    // Determine the rnn direction
    rnn_direction_ =
        bidirectional_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
    // Determine the rnn mode
    auto mode_str = OP_SINGLE_ARG(string, "rnn_mode", "");
    if (mode_str == "rnn_tanh") {
      rnn_mode_ = CUDNN_RNN_TANH;
    } else if (mode_str == "rnn_relu") {
      rnn_mode_ = CUDNN_RNN_RELU;
    } else if (mode_str == "lstm") {
      rnn_mode_ = CUDNN_LSTM;
    } else if (mode_str == "gru") {
      rnn_mode_ = CUDNN_GRU;
    } else {
      LOG(FATAL) << "Unknown RNN Mode: " << mode_str;
    }
    // Determine the rnn input mode
    auto input_mode_str = OP_SINGLE_ARG(string, "rnn_input_mode", "linear");
    if (input_mode_str == "skip") {
      rnn_input_mode_ = CUDNN_SKIP_INPUT;
    } else if (input_mode_str == "linear") {
      rnn_input_mode_ = CUDNN_LINEAR_INPUT;
    } else {
      LOG(FATAL) << "Unknown RNN InputMode: " << input_mode_str;
    }
    // Override the running phase
    SwitchToPhase(OP_SINGLE_ARG(string, "phase", ""));
    CuDNNCreateTensorDesc(&hx_desc_);
    CuDNNCreateTensorDesc(&cx_desc_);
    CuDNNCreateTensorDesc(&hy_desc_);
    CuDNNCreateTensorDesc(&cy_desc_);
    CUDNN_CHECK(cudnnCreateRNNDescriptor(&rnn_desc_));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc_));
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc_));
  }
  USE_OPERATOR_FUNCTIONS;

  virtual ~CuDNNRecurrentOpBase() {
    CuDNNDestroyTensorDesc(&hx_desc_);
    CuDNNDestroyTensorDesc(&cx_desc_);
    CuDNNDestroyTensorDesc(&hy_desc_);
    CuDNNDestroyTensorDesc(&cy_desc_);
    CUDNN_CHECK(cudnnDestroyRNNDescriptor(rnn_desc_));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc_));
    CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc_));
  }

  template <typename T>
  void SetDesc();

 public:
  float dropout_;
  unsigned long long rng_seed_;
  int64_t enable_tensor_core_;
  int64_t bidirectional_, states_initialized_;
  int64_t seq_length_, hidden_size_, num_layers_;
  vec64_t input_dims_, output_dims_, hidden_dims_;
  size_t workspace_size_, reserve_size_, states_size_;

  cudnnDataType_t compute_type_;
  cudnnRNNMode_t rnn_mode_;
  cudnnRNNDescriptor_t rnn_desc_;
  cudnnDirectionMode_t rnn_direction_;
  cudnnRNNInputMode_t rnn_input_mode_;
  cudnnDropoutDescriptor_t dropout_desc_;
  cudnnFilterDescriptor_t w_desc_;
  cudnnTensorDescriptor_t hx_desc_, cx_desc_;
  cudnnTensorDescriptor_t hy_desc_, cy_desc_;
  std::unique_ptr<CuDNNTensorDescs> x_descs_;
  std::unique_ptr<CuDNNTensorDescs> y_descs_;
};

#define USE_CUDNN_RECURRENT_FUNCTIONS                   \
  USE_OPERATOR_FUNCTIONS;                               \
  using CuDNNRecurrentOpBase<Context>::w_desc_;         \
  using CuDNNRecurrentOpBase<Context>::rnn_desc_;       \
  using CuDNNRecurrentOpBase<Context>::dropout_desc_;   \
  using CuDNNRecurrentOpBase<Context>::hx_desc_;        \
  using CuDNNRecurrentOpBase<Context>::cx_desc_;        \
  using CuDNNRecurrentOpBase<Context>::hy_desc_;        \
  using CuDNNRecurrentOpBase<Context>::cy_desc_;        \
  using CuDNNRecurrentOpBase<Context>::x_descs_;        \
  using CuDNNRecurrentOpBase<Context>::y_descs_;        \
  using CuDNNRecurrentOpBase<Context>::seq_length_;     \
  using CuDNNRecurrentOpBase<Context>::input_dims_;     \
  using CuDNNRecurrentOpBase<Context>::output_dims_;    \
  using CuDNNRecurrentOpBase<Context>::hidden_dims_;    \
  using CuDNNRecurrentOpBase<Context>::workspace_size_; \
  using CuDNNRecurrentOpBase<Context>::reserve_size_

template <class Context>
class CuDNNRecurrentOp final : public CuDNNRecurrentOpBase<Context> {
 public:
  CuDNNRecurrentOp(const OperatorDef& def, Workspace* ws)
      : CuDNNRecurrentOpBase<Context>(def, ws) {}
  USE_CUDNN_RECURRENT_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class CuDNNRecurrentGradientOp final : public CuDNNRecurrentOpBase<Context> {
 public:
  CuDNNRecurrentGradientOp(const OperatorDef& def, Workspace* ws)
      : CuDNNRecurrentOpBase<Context>(def, ws) {}
  USE_CUDNN_RECURRENT_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

#endif // USE_CUDNN

} // namespace dragon

#endif // DRAGON_OPERATORS_RECURRENT_RECURRENT_OP_CUDNN_H_
