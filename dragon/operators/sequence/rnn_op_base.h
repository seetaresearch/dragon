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

#ifndef DRAGON_OPERATORS_SEQUENCE_RNN_OP_BASE_H_
#define DRAGON_OPERATORS_SEQUENCE_RNN_OP_BASE_H_

#include "dragon/core/operator.h"

namespace dragon {

#ifdef USE_CUDNN
template <class Context>
class CuDNNRNNOpBase : public Operator<Context> {
 public:
  CuDNNRNNOpBase(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        num_layers_(OP_SINGLE_ARG(int64_t, "num_layers", 1)),
        hidden_size_(OP_SINGLE_ARG(int64_t, "hidden_size", 0)),
        bidirectional_(OP_SINGLE_ARG(int64_t, "bidirectional", 0)),
        dropout_(OP_SINGLE_ARG(float, "dropout", 0.f)) {
    SwitchToPhase(OP_SINGLE_ARG(string, "phase", ""));
    auto mode_str = OP_SINGLE_ARG(string, "rnn_mode", "");
    if (mode_str == " RNN_TANH") {
      rnn_mode_ = CUDNN_RNN_TANH;
    } else if (mode_str == "RNN_RELU") {
      rnn_mode_ = CUDNN_RNN_RELU;
    } else if (mode_str == "LSTM") {
      rnn_mode_ = CUDNN_LSTM;
    } else if (mode_str == "GRU") {
      rnn_mode_ = CUDNN_GRU;
    } else {
      LOG(FATAL) << "Unsupported RNN mode: " << mode_str;
    }
    CuDNNCreateTensorDesc(&hx_desc_);
    CuDNNCreateTensorDesc(&cx_desc_);
    CuDNNCreateTensorDesc(&hy_desc_);
    CuDNNCreateTensorDesc(&cy_desc_);
    CUDNN_CHECK(cudnnCreateRNNDescriptor(&rnn_desc_));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&weight_desc_));
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc_));
    CuDNNSetDropoutDesc(dropout_desc_, dropout_, ctx());
  }
  USE_OPERATOR_FUNCTIONS;

  ~CuDNNRNNOpBase() {
    CuDNNDestroyTensorDesc(hx_desc_);
    CuDNNDestroyTensorDesc(cx_desc_);
    CuDNNDestroyTensorDesc(hy_desc_);
    CuDNNDestroyTensorDesc(cy_desc_);
    CUDNN_CHECK(cudnnDestroyRNNDescriptor(rnn_desc_));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(weight_desc_));
    CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc_));
  }

  template <typename T>
  void SetOpDesc() {
    auto& X = Input(0);
    seqlen_ = X.dim(0);
    const auto batch_size = X.dim(1);
    const auto in_channels = X.dim(2);
    const auto num_directions = bidirectional_ > 0 ? 2 : 1;
    const auto out_channels = hidden_size_ * num_directions;
    input_dims_ = X.dims();
    output_dims_ = {seqlen_, batch_size, out_channels};
    hidden_dims_ = {num_layers_ * num_directions, batch_size, hidden_size_};

    // Set RNN.
    CUDNN_CHECK(cudnnSetRNNDescriptor_v6(
        ctx()->cudnn_handle(),
        rnn_desc_,
        hidden_size_,
        num_layers_,
        dropout_desc_,
        CUDNN_LINEAR_INPUT,
        bidirectional_ > 0 ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
        rnn_mode_,
        CUDNN_RNN_ALGO_STANDARD,
        TypeMeta::Id<T>() == TypeMeta::Id<double>() ? CUDNN_DATA_DOUBLE
                                                    : CUDNN_DATA_FLOAT));
    CUDNN_CHECK(cudnnSetRNNMatrixMathType(rnn_desc_, CuDNNGetMathType<T>()));

    // Set X and Y.
    x_descs_.reset(new CuDNNTensorDescs(seqlen_));
    y_descs_.reset(new CuDNNTensorDescs(seqlen_));
    x_descs_->Set<T>({batch_size, in_channels, 1}, {in_channels, 1, 1});
    y_descs_->Set<T>({batch_size, out_channels, 1}, {out_channels, 1, 1});

    // Set Hx, Cx, Hy and Cy.
    CuDNNSetTensorDesc<T>(hx_desc_, hidden_dims_);
    CuDNNSetTensorDesc<T>(cx_desc_, hidden_dims_);
    CuDNNSetTensorDesc<T>(hy_desc_, hidden_dims_);
    CuDNNSetTensorDesc<T>(cy_desc_, hidden_dims_);

    // Set weights.
    size_t weight_size;
    CUDNN_CHECK(cudnnGetRNNParamsSize(
        ctx()->cudnn_handle(),
        rnn_desc_,
        x_descs_->data()[0],
        &weight_size,
        CuDNNType<T>::type));
    int64_t weight_count = (int64_t)weight_size / sizeof(T);
    CHECK_EQ(Input(1).count(), weight_count)
        << "\nUnexpected count of weights: " << Input(1).count() << " vs. "
        << weight_count;
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(
        weight_desc_,
        CuDNNType<T>::type,
        CUDNN_TENSOR_NCHW,
        3,
        vec32_t({int(weight_count), 1, 1}).data()));

    // Set workspace.
    CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
        ctx()->cudnn_handle(),
        rnn_desc_,
        seqlen_,
        x_descs_->data(),
        &workspace_size_));
  }

 public:
  float dropout_;
  int64_t bidirectional_;
  int64_t seqlen_, hidden_size_, num_layers_;
  vec64_t input_dims_, output_dims_, hidden_dims_;
  size_t workspace_size_, reserve_size_;

  cudnnRNNMode_t rnn_mode_;
  cudnnRNNDescriptor_t rnn_desc_;
  cudnnFilterDescriptor_t weight_desc_;
  cudnnDropoutDescriptor_t dropout_desc_;
  cudnnTensorDescriptor_t hx_desc_, cx_desc_;
  cudnnTensorDescriptor_t hy_desc_, cy_desc_;
  std::unique_ptr<CuDNNTensorDescs> x_descs_;
  std::unique_ptr<CuDNNTensorDescs> y_descs_;
};

#define USE_CUDNN_RNN_FUNCTIONS                   \
  USE_OPERATOR_FUNCTIONS;                         \
  using CuDNNRNNOpBase<Context>::rnn_desc_;       \
  using CuDNNRNNOpBase<Context>::weight_desc_;    \
  using CuDNNRNNOpBase<Context>::dropout_desc_;   \
  using CuDNNRNNOpBase<Context>::hx_desc_;        \
  using CuDNNRNNOpBase<Context>::cx_desc_;        \
  using CuDNNRNNOpBase<Context>::hy_desc_;        \
  using CuDNNRNNOpBase<Context>::cy_desc_;        \
  using CuDNNRNNOpBase<Context>::x_descs_;        \
  using CuDNNRNNOpBase<Context>::y_descs_;        \
  using CuDNNRNNOpBase<Context>::seqlen_;         \
  using CuDNNRNNOpBase<Context>::input_dims_;     \
  using CuDNNRNNOpBase<Context>::output_dims_;    \
  using CuDNNRNNOpBase<Context>::hidden_dims_;    \
  using CuDNNRNNOpBase<Context>::workspace_size_; \
  using CuDNNRNNOpBase<Context>::reserve_size_
#endif // USE_CUDNN

} // namespace dragon

#endif // DRAGON_OPERATORS_SEQUENCE_RNN_OP_BASE_H_
