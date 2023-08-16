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

#ifndef DRAGON_OPERATORS_SEQUENCE_MHA_OP_BASE_H_
#define DRAGON_OPERATORS_SEQUENCE_MHA_OP_BASE_H_

#include "dragon/core/operator.h"

namespace dragon {

#ifdef USE_CUDNN
template <class Context>
class CuDNNMultiHeadAttnOpBase : public Operator<Context> {
 public:
  CuDNNMultiHeadAttnOpBase(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        num_heads_(OP_SINGLE_ARG(int64_t, "num_heads", 1)),
        out_channels_(OP_SINGLE_ARG(int64_t, "out_channels", 0)),
        use_bias_(OP_SINGLE_ARG(int64_t, "use_bias", 1)),
        softmax_scale_(OP_SINGLE_ARG(float, "softmax_scale", 1.f)),
        dropout_(OP_SINGLE_ARG(float, "dropout", 0.f)) {
    CHECK_EQ(out_channels_ % num_heads_, 0)
        << "\n<out_channels> must be divisible by <num_heads>";
    CUDNN_CHECK(cudnnCreateAttnDescriptor(&attn_desc_));
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc_));
    this->SwitchToPhase(OP_SINGLE_ARG(string, "phase", ""));
    attn_mode_ = CUDNN_ATTN_QUERYMAP_ALL_TO_ONE;
    attn_mode_ |= use_bias_ ? CUDNN_ATTN_ENABLE_PROJ_BIASES
                            : CUDNN_ATTN_DISABLE_PROJ_BIASES;
    CuDNNSetDropoutDesc(dropout_desc_, dropout_, this->ctx());
  }

  ~CuDNNMultiHeadAttnOpBase() {
    CUDNN_CHECK(cudnnDestroyAttnDescriptor(attn_desc_));
    CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc_));
  }

  template <typename T>
  void SetSeqDesc(cudnnSeqDataDescriptor_t desc, const vec64_t& dims) {
    vec32_t seq_dims(4, 1), seq_lens(dims[0], dims[1]);
    cudnnSeqDataAxis_t axes[4];
    axes[0] = CUDNN_SEQDATA_BATCH_DIM, axes[1] = CUDNN_SEQDATA_BEAM_DIM;
    axes[2] = CUDNN_SEQDATA_TIME_DIM, axes[3] = CUDNN_SEQDATA_VECT_DIM;
    seq_dims[CUDNN_SEQDATA_BATCH_DIM] = dims[0];
    seq_dims[CUDNN_SEQDATA_TIME_DIM] = dims[1];
    seq_dims[CUDNN_SEQDATA_VECT_DIM] = dims[2];
    CUDNN_CHECK(cudnnSetSeqDataDescriptor(
        desc,
        CuDNNTraits<T>::type,
        seq_dims.size(),
        seq_dims.data(),
        axes,
        seq_lens.size(),
        seq_lens.data(),
        nullptr));
  }

  template <typename T>
  void SetAttnDesc() {
    CUDNN_CHECK(cudnnSetAttnDescriptor(
        attn_desc_,
        attn_mode_,
        num_heads_,
        softmax_scale_,
        CuDNNTraits<T>::type,
        CuDNNTraits<T>::type,
        CuDNNGetMathType<T>(),
        dropout_desc_, // attn_drop
        dropout_desc_, // post_drop
        in_channels_, // q_size
        in_channels_, // k_size
        in_channels_, // v_size
        out_channels_ / num_heads_, // q_proj_size
        out_channels_ / num_heads_, // k_proj_size
        out_channels_ / num_heads_, // v_proj_size
        out_channels_, // o_proj_size
        q_seqlen_,
        k_seqlen_,
        batch_size_,
        1)); // beam_size
    size_t weight_size = 0;
    CUDNN_CHECK(cudnnGetMultiHeadAttnBuffers(
        this->ctx()->cudnn_handle(),
        this->attn_desc_,
        &weight_size,
        &train_workspace_size_,
        &reserve_size_));
    CUDNN_CHECK(cudnnGetMultiHeadAttnBuffers(
        this->ctx()->cudnn_handle(),
        this->attn_desc_,
        &weight_size,
        &infer_workspace_size_,
        nullptr));
    // Q/K/V weight layout: [in_channels, num_heads, proj_size]
    //     O weight layout: [num_heads, proj_size, out_channels]
    CHECK_EQ(weight_size / sizeof(T), weight_count_)
        << "\nUnexpected count of weights: " << weight_count_ << " vs. "
        << weight_size / sizeof(T);
  }

 protected:
  int64_t batch_size_, q_seqlen_, k_seqlen_;
  int64_t num_heads_, in_channels_, out_channels_;
  int64_t weight_count_, use_bias_;
  float softmax_scale_, dropout_;

  cudnnAttnDescriptor_t attn_desc_;
  cudnnDropoutDescriptor_t dropout_desc_;
  size_t train_workspace_size_, infer_workspace_size_;
  size_t reserve_size_;
  unsigned int attn_mode_;
};

template <class Context>
class CuDNNMultiHeadSelfAttnOpBase : public CuDNNMultiHeadAttnOpBase<Context> {
 public:
  CuDNNMultiHeadSelfAttnOpBase(const OperatorDef& def, Workspace* ws)
      : CuDNNMultiHeadAttnOpBase<Context>(def, ws) {
    CUDNN_CHECK(cudnnCreateSeqDataDescriptor(&input_desc_));
    CUDNN_CHECK(cudnnCreateSeqDataDescriptor(&output_desc_));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CuDNNMultiHeadSelfAttnOpBase() {
    CUDNN_CHECK(cudnnDestroySeqDataDescriptor(input_desc_));
    CUDNN_CHECK(cudnnDestroySeqDataDescriptor(output_desc_));
  }

  template <typename T>
  void SetOpDesc() {
    auto& X = Input(0);
    this->batch_size_ = X.dim(0);
    this->q_seqlen_ = this->k_seqlen_ = X.dim(1);
    this->in_channels_ = X.dim(2);
    this->weight_count_ = this->in_channels_ * this->out_channels_ * 4;
    this->weight_count_ += this->use_bias_ > 0 ? this->out_channels_ * 4 : 0;
    input_dims_ = {this->batch_size_, this->q_seqlen_, this->in_channels_},
    output_dims_ = {this->batch_size_, this->q_seqlen_, this->out_channels_};
    this->template SetAttnDesc<T>();
    this->template SetSeqDesc<T>(input_desc_, input_dims_);
    this->template SetSeqDesc<T>(output_desc_, output_dims_);
  }

 protected:
  vec64_t input_dims_, output_dims_;
  cudnnSeqDataDescriptor_t input_desc_, output_desc_;
};
#endif // USE_CUDNN

#ifdef USE_MLU
template <class Context>
class CNNLMultiHeadAttnOpBase : public Operator<Context> {
 public:
  CNNLMultiHeadAttnOpBase(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        num_heads_(OP_SINGLE_ARG(int64_t, "num_heads", 1)),
        out_channels_(OP_SINGLE_ARG(int64_t, "out_channels", 0)),
        use_bias_(OP_SINGLE_ARG(int64_t, "use_bias", 1)),
        softmax_scale_(OP_SINGLE_ARG(float, "softmax_scale", 1.f)),
        dropout_(OP_SINGLE_ARG(float, "dropout", 0.f)) {
    CHECK_EQ(out_channels_ % num_heads_, 0)
        << "\n<out_channels> must be divisible by <num_heads>";
    CNNL_CHECK(cnnlCreateMultiHeadAttnDescriptor(&attn_desc_));
    this->SwitchToPhase(OP_SINGLE_ARG(string, "phase", ""));
  }

  ~CNNLMultiHeadAttnOpBase() {
    CNNL_CHECK(cnnlDestroyMultiHeadAttnDescriptor(attn_desc_));
  }

  template <typename T>
  void SetSeqDesc(cnnlSeqDataDescriptor_t desc, const vec64_t& dims) {
    vec32_t seq_dims({dims.begin(), dims.end()});
    if (seq_dims.size() == 3) seq_dims.insert(seq_dims.begin() + 1, 1);
    CNNL_CHECK(cnnlSetSeqDataDescriptor(
        desc,
        CNNL_SEQDATA_NBTC,
        CNNLGetDataType<T>(),
        seq_dims.size(),
        seq_dims.data(),
        0,
        nullptr,
        nullptr));
  }

  template <typename T>
  void SetAttnDesc() {
    CNNL_CHECK(cnnlSetMultiHeadAttnDescriptor(
        attn_desc_,
        use_bias_ ? ALL_TO_ONE_BIAS : ALL_TO_ONE_NOBIAS,
        num_heads_,
        softmax_scale_,
        CNNLGetDataType<T>(),
        CNNLGetDataType<T>(),
        dropout_, // attn_drop
        dropout_, // post_drop
        in_channels_, // q_size
        in_channels_, // k_size
        in_channels_, // v_size
        out_channels_ / num_heads_, // q_proj_size
        out_channels_ / num_heads_, // k_proj_size
        out_channels_ / num_heads_, // v_proj_size
        out_channels_, // o_proj_size
        q_seqlen_,
        k_seqlen_,
        batch_size_,
        1)); // beam_size
    size_t weight_size = 0;
    CNNL_CHECK(cnnlGetMultiHeadAttnBuffers(
        this->ctx()->cnnl_handle(),
        this->attn_desc_,
        &weight_size,
        &train_workspace_size_,
        &reserve_size_));
    CNNL_CHECK(cnnlGetMultiHeadAttnBuffers(
        this->ctx()->cnnl_handle(),
        this->attn_desc_,
        &weight_size,
        &infer_workspace_size_,
        nullptr));
    // Q/K/V weight layout: [num_heads, proj_size, in_channels]
    //     O weight layout: [num_heads, out_channels, proj_size]
    CHECK_EQ(weight_size / sizeof(T), weight_count_)
        << "\nUnexpected count of weights: " << weight_count_ << " vs. "
        << weight_size / sizeof(T);
  }

 protected:
  int64_t batch_size_, q_seqlen_, k_seqlen_;
  int64_t num_heads_, in_channels_, out_channels_;
  int64_t weight_count_, use_bias_;
  float softmax_scale_, dropout_;

  cnnlMultiHeadAttnDescriptor_t attn_desc_;
  size_t train_workspace_size_, infer_workspace_size_;
  size_t reserve_size_;
};

template <class Context>
class CNNLMultiHeadSelfAttnOpBase : public CNNLMultiHeadAttnOpBase<Context> {
 public:
  CNNLMultiHeadSelfAttnOpBase(const OperatorDef& def, Workspace* ws)
      : CNNLMultiHeadAttnOpBase<Context>(def, ws) {
    CNNLCreateTensorDesc(&attn_mask_desc_);
    CNNL_CHECK(cnnlCreateSeqDataDescriptor(&input_desc_));
    CNNL_CHECK(cnnlCreateSeqDataDescriptor(&output_desc_));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLMultiHeadSelfAttnOpBase() {
    CNNLDestroyTensorDesc(attn_mask_desc_);
    CNNL_CHECK(cnnlDestroySeqDataDescriptor(input_desc_));
    CNNL_CHECK(cnnlDestroySeqDataDescriptor(output_desc_));
  }

  template <typename T>
  void SetOpDesc() {
    auto& X = Input(0);
    this->batch_size_ = X.dim(0);
    this->q_seqlen_ = this->k_seqlen_ = X.dim(1);
    this->in_channels_ = X.dim(2);
    this->weight_count_ = this->in_channels_ * this->out_channels_ * 4;
    this->weight_count_ += this->use_bias_ > 0 ? this->out_channels_ * 4 : 0;
    input_dims_ = {this->batch_size_, this->q_seqlen_, this->in_channels_},
    output_dims_ = {this->batch_size_, this->q_seqlen_, this->out_channels_};
    this->template SetAttnDesc<T>();
    this->template SetSeqDesc<T>(input_desc_, input_dims_);
    this->template SetSeqDesc<T>(output_desc_, output_dims_);
    if (InputSize() > 2 && Input(2).has_name()) {
      CNNLSetTensorDesc<T>(attn_mask_desc_, Input(2).dims());
    }
  }

 protected:
  vec64_t input_dims_, output_dims_;
  cnnlTensorDescriptor_t attn_mask_desc_;
  cnnlSeqDataDescriptor_t input_desc_, output_desc_;
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_SEQUENCE_MHA_OP_BASE_H_
