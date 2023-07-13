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

#ifndef DRAGON_OPERATORS_VISION_CONV_OP_IMPL_CNNL_H_
#define DRAGON_OPERATORS_VISION_CONV_OP_IMPL_CNNL_H_

#ifdef USE_MLU

#include "dragon/core/context_mlu.h"
#include "dragon/core/workspace.h"

namespace dragon {

template <typename Algo>
class CNNLConvOpImpl {};

template <typename Algo>
class CNNLDeconvOpImpl {};

template <typename T>
void CNNLSetConvDesc(
    cnnlConvolutionDescriptor_t conv_desc,
    const vec64_t& pads_begin,
    const vec64_t& pads_end,
    const vec64_t& strides,
    const vec64_t& dilations,
    const int64_t group) {
  vec64_t strides_v2(strides);
  vec64_t dilations_v2(dilations);
  if (strides.size() == 1) {
    strides_v2.push_back(1);
    dilations_v2.push_back(1);
  }
  vec64_t pads(strides_v2.size() * 2, 0);
  for (int i = 0; i < pads_begin.size(); ++i) {
    pads[i * 2] = pads_begin[i];
    pads[i * 2 + 1] = pads_end[i];
  }
  CNNL_CHECK(cnnlSetConvolutionDescriptor(
      conv_desc,
      strides_v2.size() + 2,
      vec32_t({pads.begin(), pads.end()}).data(),
      vec32_t({strides_v2.begin(), strides_v2.end()}).data(),
      vec32_t({dilations_v2.begin(), dilations_v2.end()}).data(),
      group,
      CNNLGetDataType<T>()));
}

template <typename T>
void CNNLSetDeconvDesc(
    cnnlDeconvolutionDescriptor_t conv_desc,
    const vec64_t& pads_begin,
    const vec64_t& pads_end,
    const vec64_t& strides,
    const vec64_t& dilations,
    const int64_t group) {
  vec64_t strides_v2(strides);
  vec64_t dilations_v2(dilations);
  if (strides.size() == 1) {
    strides_v2.push_back(1);
    dilations_v2.push_back(1);
  }
  vec64_t pads(strides_v2.size() * 2, 0);
  for (int i = 0; i < pads_begin.size(); ++i) {
    pads[i * 2] = pads_begin[i];
    pads[i * 2 + 1] = pads_end[i];
  }
  CNNL_CHECK(cnnlSetDeconvolutionDescriptor(
      conv_desc,
      strides_v2.size() + 2,
      vec32_t({pads.begin(), pads.end()}).data(),
      vec32_t({strides_v2.begin(), strides_v2.end()}).data(),
      vec32_t({dilations_v2.begin(), dilations_v2.end()}).data(),
      group,
      CNNLGetDataType<T>()));
}

template <>
class CNNLConvOpImpl<cnnlConvolutionForwardAlgo_t> {
 public:
  CNNLConvOpImpl() : workspace_size_(0) {
    CNNLCreateTensorDesc(&X_desc_);
    CNNLCreateTensorDesc(&W_desc_);
    CNNLCreateTensorDesc(&B_desc_);
    CNNLCreateTensorDesc(&Y_desc_);
    CNNL_CHECK(cnnlCreateConvolutionDescriptor(&conv_desc_));
  }

  ~CNNLConvOpImpl() {
    CNNLDestroyTensorDesc(X_desc_);
    CNNLDestroyTensorDesc(W_desc_);
    CNNLDestroyTensorDesc(B_desc_);
    CNNLDestroyTensorDesc(Y_desc_);
    CNNL_CHECK(cnnlDestroyConvolutionDescriptor(conv_desc_));
  }

  template <typename T>
  void Setup(
      const vec64_t& pads_begin,
      const vec64_t& pads_end,
      const vec64_t& strides,
      const vec64_t& dilations,
      const int64_t group,
      const string& data_format,
      const vec64_t& X_dims,
      const vec64_t& W_dims,
      const vec64_t& Y_dims,
      MLUContext* ctx) {
    CNNLSetConvDesc<T>(
        conv_desc_, pads_begin, pads_end, strides, dilations, group);
    auto W_dims_v2 = W_dims;
    if (data_format == "NHWC") {
      W_dims_v2.push_back(W_dims[1]);
      W_dims_v2.erase(W_dims_v2.begin() + 1);
    }
    CNNLSetTensorDesc<T>(X_desc_, X_dims, data_format);
    CNNLSetTensorDesc<T>(W_desc_, W_dims_v2, data_format);
    CNNLSetTensorDesc<T>(B_desc_, vec64_t({W_dims[0]}), data_format);
    CNNLSetTensorDesc<T>(Y_desc_, Y_dims, data_format);
    CNNL_CHECK(cnnlGetConvolutionForwardAlgorithm(
        ctx->cnnl_handle(),
        conv_desc_,
        X_desc_,
        W_desc_,
        Y_desc_,
        CNNL_CONVOLUTION_FWD_FASTEST,
        &algo_));
    CNNL_CHECK(cnnlGetConvolutionForwardWorkspaceSize(
        ctx->cnnl_handle(),
        X_desc_,
        W_desc_,
        Y_desc_,
        B_desc_,
        conv_desc_,
        algo_,
        &workspace_size_));
  }

  template <typename T>
  void Compute(const T* X, const T* W, const T* B, T* Y, MLUContext* ctx) {
    CNNL_CHECK(cnnlConvolutionForward(
        ctx->cnnl_handle(),
        conv_desc_,
        algo_,
        nullptr, // alpha
        X_desc_,
        X,
        W_desc_,
        W,
        B == nullptr ? nullptr : B_desc_,
        B,
        ctx->workspace()->data<MLUContext>(workspace_size_),
        workspace_size_,
        nullptr, // beta
        Y_desc_,
        Y));
  }

  size_t workspace_size_;
  cnnlConvolutionForwardAlgo_t algo_;
  cnnlConvolutionDescriptor_t conv_desc_;
  cnnlTensorDescriptor_t X_desc_, W_desc_, B_desc_, Y_desc_;
};

template <>
class CNNLConvOpImpl<cnnlConvolutionBwdDataAlgo_t> {
 public:
  CNNLConvOpImpl() : workspace_size_(0) {
    CNNLCreateTensorDesc(&X_desc_);
    CNNLCreateTensorDesc(&W_desc_);
    CNNLCreateTensorDesc(&Y_desc_);
    CNNL_CHECK(cnnlCreateConvolutionDescriptor(&conv_desc_));
  }

  ~CNNLConvOpImpl() {
    CNNLDestroyTensorDesc(X_desc_);
    CNNLDestroyTensorDesc(W_desc_);
    CNNLDestroyTensorDesc(Y_desc_);
    CNNL_CHECK(cnnlDestroyConvolutionDescriptor(conv_desc_));
  }

  template <typename T>
  void Setup(
      const vec64_t& pads_begin,
      const vec64_t& pads_end,
      const vec64_t& strides,
      const vec64_t& dilations,
      const int64_t group,
      const string& data_format,
      const vec64_t& X_dims,
      const vec64_t& W_dims,
      const vec64_t& Y_dims,
      MLUContext* ctx) {
    CNNLSetConvDesc<T>(
        conv_desc_, pads_begin, pads_end, strides, dilations, group);
    auto W_dims_v2 = W_dims;
    if (data_format == "NHWC") {
      W_dims_v2.push_back(W_dims[1]);
      W_dims_v2.erase(W_dims_v2.begin() + 1);
    }
    CNNLSetTensorDesc<T>(X_desc_, X_dims, data_format);
    CNNLSetTensorDesc<T>(W_desc_, W_dims_v2, data_format);
    CNNLSetTensorDesc<T>(Y_desc_, Y_dims, data_format);
    CNNL_CHECK(cnnlGetConvolutionBackwardDataAlgorithm(
        ctx->cnnl_handle(),
        W_desc_,
        Y_desc_,
        conv_desc_,
        X_desc_,
        CNNL_CONVOLUTION_BWD_DATA_FASTEST,
        &algo_));
    CNNL_CHECK(cnnlGetConvolutionBackwardDataWorkspaceSize(
        ctx->cnnl_handle(),
        W_desc_,
        Y_desc_,
        conv_desc_,
        X_desc_,
        algo_,
        &workspace_size_));
  }

  template <typename T>
  void Compute(const T* dY, const T* W, T* dX, MLUContext* ctx) {
    CNNL_CHECK(cnnlConvolutionBackwardData(
        ctx->cnnl_handle(),
        nullptr, // alpha
        W_desc_,
        W,
        Y_desc_,
        dY,
        conv_desc_,
        algo_,
        ctx->workspace()->data<MLUContext>(workspace_size_),
        workspace_size_,
        nullptr, // beta
        X_desc_,
        dX));
  }

  size_t workspace_size_;
  cnnlConvolutionBwdDataAlgo_t algo_;
  cnnlConvolutionDescriptor_t conv_desc_;
  cnnlTensorDescriptor_t X_desc_, W_desc_, Y_desc_;
};

template <>
class CNNLConvOpImpl<cnnlConvolutionBwdFilterAlgo_t> {
 public:
  CNNLConvOpImpl() : workspace_size_(0) {
    CNNLCreateTensorDesc(&X_desc_);
    CNNLCreateTensorDesc(&W_desc_);
    CNNLCreateTensorDesc(&Y_desc_);
    CNNL_CHECK(cnnlCreateConvolutionDescriptor(&conv_desc_));
  }

  ~CNNLConvOpImpl() {
    CNNLDestroyTensorDesc(X_desc_);
    CNNLDestroyTensorDesc(W_desc_);
    CNNLDestroyTensorDesc(Y_desc_);
    CNNL_CHECK(cnnlDestroyConvolutionDescriptor(conv_desc_));
  }

  template <typename T>
  void Setup(
      const vec64_t& pads_begin,
      const vec64_t& pads_end,
      const vec64_t& strides,
      const vec64_t& dilations,
      const int64_t group,
      const string& data_format,
      const vec64_t& X_dims,
      const vec64_t& W_dims,
      const vec64_t& Y_dims,
      MLUContext* ctx) {
    CNNLSetConvDesc<T>(
        conv_desc_, pads_begin, pads_end, strides, dilations, group);
    auto W_dims_v2 = W_dims;
    if (data_format == "NHWC") {
      W_dims_v2.push_back(W_dims[1]);
      W_dims_v2.erase(W_dims_v2.begin() + 1);
    }
    CNNLSetTensorDesc<T>(X_desc_, X_dims, data_format);
    CNNLSetTensorDesc<T>(W_desc_, W_dims_v2, data_format);
    CNNLSetTensorDesc<T>(Y_desc_, Y_dims, data_format);
    CNNL_CHECK(cnnlGetConvolutionBackwardFilterAlgorithm(
        ctx->cnnl_handle(),
        conv_desc_,
        X_desc_,
        Y_desc_,
        W_desc_,
        CNNL_CONVOLUTION_BWD_FILTER_FASTEST,
        &algo_));
    CNNL_CHECK(cnnlGetConvolutionBackwardFilterWorkspaceSize(
        ctx->cnnl_handle(),
        X_desc_,
        Y_desc_,
        W_desc_,
        conv_desc_,
        algo_,
        &workspace_size_));
  }

  template <typename T>
  void Compute(const T* dY, const T* X, T* dW, MLUContext* ctx) {
    CNNL_CHECK(cnnlConvolutionBackwardFilter(
        ctx->cnnl_handle(),
        nullptr, // alpha
        X_desc_,
        X,
        Y_desc_,
        dY,
        conv_desc_,
        algo_,
        ctx->workspace()->data<MLUContext>(workspace_size_),
        workspace_size_,
        nullptr, // beta
        W_desc_,
        dW));
  }

  size_t workspace_size_;
  cnnlConvolutionBwdFilterAlgo_t algo_;
  cnnlConvolutionDescriptor_t conv_desc_;
  cnnlTensorDescriptor_t X_desc_, W_desc_, Y_desc_;
};

template <>
class CNNLDeconvOpImpl<cnnlDeconvolutionAlgo_t> {
 public:
  CNNLDeconvOpImpl() : workspace_size_(0) {
    CNNLCreateTensorDesc(&X_desc_);
    CNNLCreateTensorDesc(&W_desc_);
    CNNLCreateTensorDesc(&B_desc_);
    CNNLCreateTensorDesc(&Y_desc_);
    CNNL_CHECK(cnnlCreateDeconvolutionDescriptor(&conv_desc_));
  }

  ~CNNLDeconvOpImpl() {
    CNNLDestroyTensorDesc(X_desc_);
    CNNLDestroyTensorDesc(W_desc_);
    CNNLDestroyTensorDesc(B_desc_);
    CNNLDestroyTensorDesc(Y_desc_);
    CNNL_CHECK(cnnlDestroyDeconvolutionDescriptor(conv_desc_));
  }

  template <typename T>
  void Setup(
      const vec64_t& pads_begin,
      const vec64_t& pads_end,
      const vec64_t& strides,
      const vec64_t& dilations,
      const int64_t group,
      const string& data_format,
      const vec64_t& X_dims,
      const vec64_t& W_dims,
      const vec64_t& Y_dims,
      MLUContext* ctx) {
    CNNLSetDeconvDesc<T>(
        conv_desc_, pads_begin, pads_end, strides, dilations, group);
    auto W_dims_v2 = W_dims;
    auto B_dims_v2 = vec64_t(W_dims.size(), 1);
    if (data_format == "NHWC") {
      W_dims_v2.push_back(W_dims[1]);
      W_dims_v2.erase(W_dims_v2.begin() + 1);
      B_dims_v2.back() = W_dims[1];
    } else {
      B_dims_v2[1] = W_dims[1];
    }
    CNNLSetTensorDesc<T>(X_desc_, X_dims, data_format);
    CNNLSetTensorDesc<T>(W_desc_, W_dims_v2, data_format);
    CNNLSetTensorDesc<T>(B_desc_, B_dims_v2, data_format);
    CNNLSetTensorDesc<T>(Y_desc_, Y_dims, data_format);
    CNNL_CHECK(cnnlGetDeconvolutionAlgorithm_v2(
        ctx->cnnl_handle(),
        X_desc_,
        W_desc_,
        B_desc_,
        conv_desc_,
        Y_desc_,
        CNNL_CONVOLUTION_BWD_DATA_FASTEST,
        &algo_));
    CNNL_CHECK(cnnlGetDeconvolutionWorkspaceSize_v2(
        ctx->cnnl_handle(),
        X_desc_,
        W_desc_,
        B_desc_,
        conv_desc_,
        Y_desc_,
        algo_,
        &workspace_size_));
  }

  template <typename T>
  void Compute(const T* X, const T* W, const T* B, T* Y, MLUContext* ctx) {
    CNNL_CHECK(cnnlDeconvolution(
        ctx->cnnl_handle(),
        nullptr, // alpha
        X_desc_,
        X,
        W_desc_,
        W,
        B == nullptr ? nullptr : B_desc_,
        B,
        conv_desc_,
        algo_,
        ctx->workspace()->data<MLUContext>(workspace_size_),
        workspace_size_,
        nullptr, // beta
        Y_desc_,
        Y));
  }

  size_t workspace_size_;
  cnnlDeconvolutionAlgo_t algo_;
  cnnlDeconvolutionDescriptor_t conv_desc_;
  cnnlTensorDescriptor_t X_desc_, W_desc_, B_desc_, Y_desc_;
};

} // namespace dragon

#endif // USE_MLU

#endif // DRAGON_OPERATORS_VISION_CONV_OP_IMPL_CNNL_H_
