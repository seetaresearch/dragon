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

#ifndef DRAGON_OPERATORS_VISION_CONV_OP_IMPL_CUDNN_H_
#define DRAGON_OPERATORS_VISION_CONV_OP_IMPL_CUDNN_H_

#ifdef USE_CUDNN

#include "dragon/core/context_cuda.h"
#include "dragon/core/workspace.h"
#include "dragon/operators/vision/conv_op_algo.h"

namespace dragon {

template <typename Algo>
class CuDNNConvOpImpl {};

template <typename T>
void CuDNNSetConvDesc(
    cudnnConvolutionDescriptor_t conv_desc,
    const vec64_t& pads_begin,
    const vec64_t& strides,
    const vec64_t& dilations,
    const int64_t group) {
  const int num_axes = strides.size();
  if (num_axes == 1 || num_axes == 2) {
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pads_begin[0],
        num_axes == 1 ? 0 : pads_begin[1],
        strides[0],
        num_axes == 1 ? 1 : strides[1],
        dilations[0],
        num_axes == 1 ? 1 : dilations[1],
        CUDNN_CROSS_CORRELATION,
        TypeMeta::Id<T>() == TypeMeta::Id<double>() ? CUDNN_DATA_DOUBLE
                                                    : CUDNN_DATA_FLOAT));
  } else {
    CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
        conv_desc,
        num_axes,
        vec32_t{pads_begin.begin(), pads_begin.end()}.data(),
        vec32_t{strides.begin(), strides.end()}.data(),
        vec32_t{dilations.begin(), dilations.end()}.data(),
        CUDNN_CROSS_CORRELATION,
        TypeMeta::Id<T>() == TypeMeta::Id<double>() ? CUDNN_DATA_DOUBLE
                                                    : CUDNN_DATA_FLOAT));
  }
  CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, group));
  CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CuDNNGetMathType<T>()));
}

template <typename T>
void CuDNNSetConvFilterDesc(
    cudnnFilterDescriptor_t filter_desc,
    const vec64_t& filter_dims,
    const string& data_format) {
  const int num_axes = filter_dims.size() - 2;
  if (num_axes == 1 || num_axes == 2) {
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        filter_desc,
        CuDNNTraits<T>::type,
        data_format == "NCHW" ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC,
        filter_dims[0],
        filter_dims[1],
        filter_dims[2],
        num_axes == 1 ? 1 : filter_dims[3]));
  } else {
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(
        filter_desc,
        CuDNNTraits<T>::type,
        data_format == "NCHW" ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC,
        filter_dims.size(),
        vec32_t{filter_dims.begin(), filter_dims.end()}.data()));
  }
}

template <>
class CuDNNConvOpImpl<cudnnConvolutionFwdAlgo_t> {
 public:
  using AlgoT = cudnnConvolutionFwdAlgo_t;
  using AlgoWithCost = std::tuple<AlgoT, float>;

  CuDNNConvOpImpl() : workspace_size_(0) {
    CuDNNCreateTensorDesc(&X_desc_);
    CuDNNCreateTensorDesc(&B_desc_);
    CuDNNCreateTensorDesc(&Y_desc_);
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&W_desc_));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
  }

  ~CuDNNConvOpImpl() {
    CuDNNDestroyTensorDesc(X_desc_);
    CuDNNDestroyTensorDesc(B_desc_);
    CuDNNDestroyTensorDesc(Y_desc_);
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(W_desc_));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }

  template <typename T>
  void Setup(
      const vec64_t& pads_begin,
      const vec64_t& strides,
      const vec64_t& dilations,
      const int64_t group,
      const string& data_format,
      const vec64_t& X_dims,
      const vec64_t& W_dims,
      const vec64_t& Y_dims,
      CUDAContext* ctx) {
    const bool input_changed = (X_dims != X_dims_);
    const bool filter_changed = (W_dims != W_dims_);
    CuDNNSetConvDesc<T>(conv_desc_, pads_begin, strides, dilations, group);
    if (!input_changed && !filter_changed) return;
    if (input_changed) {
      CuDNNSetTensorDesc<T>(X_desc_, X_dims_ = X_dims, data_format);
      CuDNNSetTensorDesc<T>(Y_desc_, Y_dims, data_format);
    }
    if (filter_changed) {
      CuDNNSetConvFilterDesc<T>(W_desc_, W_dims_ = W_dims, data_format);
      CuDNNSetBiasDesc<T>(B_desc_, X_dims.size(), W_dims[0], data_format);
    }
    workspace_size_ = SIZE_MAX;
    workspace_maxsize_ = CUDNN_CONV_WORKSPACE_LIMIT_BYTES;
    if (CUDAContext::objects().cudnn_benchmark_) {
      exhaustive_search_ = true;
      return;
    }
    exhaustive_search_ = false;
    if (CUDAContext::objects().cudnn_deterministic_) {
      algo_ = ConvAlgoSearch<AlgoT>().get_deterministic();
      return;
    }
    auto fn = [&]() {
      return AlgoWithCost(
          ConvAlgoSearch<AlgoT>().get(
              ctx->cudnn_handle(),
              X_desc_,
              W_desc_,
              conv_desc_,
              Y_desc_,
              workspace_maxsize_),
          0.f);
    };
    auto result = algo_cache_.get(X_dims, W_dims, TypeMeta::Id<T>(), fn);
    algo_ = std::get<0>(result);
  }

  template <typename T>
  void Compute(const T* X, const T* W, const T* B, T* Y, CUDAContext* ctx) {
    if (exhaustive_search_) {
      exhaustive_search_ = false;
      auto fn = [&]() {
        return ConvAlgoSearch<AlgoT>().find(
            ctx->cudnn_handle(),
            X_desc_,
            W_desc_,
            conv_desc_,
            Y_desc_,
            workspace_maxsize_,
            X,
            W,
            Y,
            ctx->workspace()->template data<CUDAContext>(workspace_maxsize_));
      };
      auto result = algo_cache_.get(X_dims_, W_dims_, TypeMeta::Id<T>(), fn);
      algo_ = std::get<0>(result);
    }
    if (workspace_size_ == SIZE_MAX) {
      auto algo_status = CUDNN_STATUS_SUCCESS;
      for (int step = 0; step < 2; ++step) {
        algo_status = cudnnGetConvolutionForwardWorkspaceSize(
            ctx->cudnn_handle(),
            X_desc_,
            W_desc_,
            conv_desc_,
            Y_desc_,
            algo_,
            &workspace_size_);
        if (algo_status == CUDNN_STATUS_SUCCESS) break;
        algo_ = ConvAlgoSearch<AlgoT>().get_default();
      }
      CUDNN_CHECK(algo_status);
    }
    CUDNN_CHECK(cudnnConvolutionForward(
        ctx->cudnn_handle(),
        CuDNNTraits<T>::one,
        X_desc_,
        X,
        W_desc_,
        W,
        conv_desc_,
        algo_,
        ctx->workspace()->data<CUDAContext>(workspace_size_),
        workspace_size_,
        CuDNNTraits<T>::zero,
        Y_desc_,
        Y));
    if (B != nullptr) {
      CUDNN_CHECK(cudnnAddTensor(
          ctx->cudnn_handle(),
          CuDNNTraits<T>::one,
          B_desc_,
          B,
          CuDNNTraits<T>::one,
          Y_desc_,
          Y));
    }
  }

  vec64_t X_dims_, W_dims_;
  size_t workspace_size_, workspace_maxsize_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnFilterDescriptor_t W_desc_;
  cudnnTensorDescriptor_t X_desc_, Y_desc_, B_desc_;
  AlgoT algo_;
  ConvAlgoCache<AlgoWithCost> algo_cache_;
  bool exhaustive_search_;
};

template <>
class CuDNNConvOpImpl<cudnnConvolutionBwdDataAlgo_t> {
 public:
  using AlgoT = cudnnConvolutionBwdDataAlgo_t;
  using AlgoWithCost = std::tuple<AlgoT, float>;

  CuDNNConvOpImpl() : workspace_size_(0) {
    CuDNNCreateTensorDesc(&X_desc_);
    CuDNNCreateTensorDesc(&B_desc_);
    CuDNNCreateTensorDesc(&Y_desc_);
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&W_desc_));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
  }

  ~CuDNNConvOpImpl() {
    CuDNNDestroyTensorDesc(X_desc_);
    CuDNNDestroyTensorDesc(B_desc_);
    CuDNNDestroyTensorDesc(Y_desc_);
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(W_desc_));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }

  template <typename T>
  void Setup(
      const vec64_t& pads_begin,
      const vec64_t& strides,
      const vec64_t& dilations,
      const int64_t group,
      const string& data_format,
      const vec64_t& X_dims,
      const vec64_t& W_dims,
      const vec64_t& Y_dims,
      CUDAContext* ctx) {
    const bool input_changed = (X_dims != X_dims_);
    const bool filter_changed = (W_dims != W_dims_);
    CuDNNSetConvDesc<T>(conv_desc_, pads_begin, strides, dilations, group);
    if (!input_changed && !filter_changed) return;
    if (input_changed) {
      CuDNNSetTensorDesc<T>(X_desc_, X_dims_ = X_dims, data_format);
      CuDNNSetTensorDesc<T>(Y_desc_, Y_dims, data_format);
    }
    if (filter_changed) {
      CuDNNSetConvFilterDesc<T>(W_desc_, W_dims_ = W_dims, data_format);
    }
    workspace_size_ = SIZE_MAX;
    workspace_maxsize_ = CUDNN_CONV_WORKSPACE_LIMIT_BYTES;
    if (CUDAContext::objects().cudnn_benchmark_) {
      exhaustive_search_ = true;
      return;
    }
    exhaustive_search_ = false;
    if (CUDAContext::objects().cudnn_deterministic_) {
      algo_ = ConvAlgoSearch<AlgoT>().get_deterministic();
      return;
    }
    auto fn = [&]() {
      return AlgoWithCost(
          ConvAlgoSearch<AlgoT>().get(
              ctx->cudnn_handle(),
              W_desc_,
              Y_desc_,
              conv_desc_,
              X_desc_,
              workspace_maxsize_),
          0.f);
    };
    auto result = algo_cache_.get(X_dims, W_dims, TypeMeta::Id<T>(), fn);
    algo_ = std::get<0>(result);
  }

  template <typename T>
  void Compute(const T* dY, const T* W, T* dX, CUDAContext* ctx) {
    if (exhaustive_search_) {
      exhaustive_search_ = false;
      auto fn = [&]() {
        return ConvAlgoSearch<AlgoT>().find(
            ctx->cudnn_handle(),
            W_desc_,
            Y_desc_,
            conv_desc_,
            X_desc_,
            workspace_maxsize_,
            W,
            dY,
            dX,
            ctx->workspace()->data<CUDAContext>(workspace_maxsize_));
      };
      auto result = algo_cache_.get(X_dims_, W_dims_, TypeMeta::Id<T>(), fn);
      algo_ = std::get<0>(result);
    }
    if (workspace_size_ == SIZE_MAX) {
      auto algo_status = CUDNN_STATUS_SUCCESS;
      for (int step = 0; step < 2; ++step) {
        algo_status = cudnnGetConvolutionBackwardDataWorkspaceSize(
            ctx->cudnn_handle(),
            W_desc_,
            Y_desc_,
            conv_desc_,
            X_desc_,
            algo_,
            &workspace_size_);
        if (algo_status == CUDNN_STATUS_SUCCESS) break;
        algo_ = ConvAlgoSearch<AlgoT>().get_default();
      }
      CUDNN_CHECK(algo_status);
    }
    CUDNN_CHECK(cudnnConvolutionBackwardData(
        ctx->cudnn_handle(),
        CuDNNTraits<T>::one,
        W_desc_,
        W,
        Y_desc_,
        dY,
        conv_desc_,
        algo_,
        ctx->workspace()->data<CUDAContext>(workspace_size_),
        workspace_size_,
        CuDNNTraits<T>::zero,
        X_desc_,
        dX));
  }

  vec64_t X_dims_, W_dims_;
  size_t workspace_size_, workspace_maxsize_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnFilterDescriptor_t W_desc_;
  cudnnTensorDescriptor_t X_desc_, Y_desc_, B_desc_;
  AlgoT algo_;
  ConvAlgoCache<AlgoWithCost> algo_cache_;
  bool exhaustive_search_;
};

template <>
class CuDNNConvOpImpl<cudnnConvolutionBwdFilterAlgo_t> {
 public:
  using AlgoT = cudnnConvolutionBwdFilterAlgo_t;
  using AlgoWithCost = std::tuple<AlgoT, float>;

  CuDNNConvOpImpl() : workspace_size_(0) {
    CuDNNCreateTensorDesc(&X_desc_);
    CuDNNCreateTensorDesc(&B_desc_);
    CuDNNCreateTensorDesc(&Y_desc_);
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&W_desc_));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
  }

  ~CuDNNConvOpImpl() {
    CuDNNDestroyTensorDesc(X_desc_);
    CuDNNDestroyTensorDesc(B_desc_);
    CuDNNDestroyTensorDesc(Y_desc_);
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(W_desc_));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }

  template <typename T>
  void Setup(
      const vec64_t& pads_begin,
      const vec64_t& strides,
      const vec64_t& dilations,
      const int64_t group,
      const string& data_format,
      const vec64_t& X_dims,
      const vec64_t& W_dims,
      const vec64_t& Y_dims,
      CUDAContext* ctx) {
    const bool input_changed = (X_dims != X_dims_);
    const bool filter_changed = (W_dims != W_dims_);
    CuDNNSetConvDesc<T>(conv_desc_, pads_begin, strides, dilations, group);
    if (!input_changed && !filter_changed) return;
    if (input_changed) {
      CuDNNSetTensorDesc<T>(X_desc_, X_dims_ = X_dims, data_format);
      CuDNNSetTensorDesc<T>(Y_desc_, Y_dims, data_format);
    }
    if (filter_changed) {
      CuDNNSetConvFilterDesc<T>(W_desc_, W_dims_ = W_dims, data_format);
      CuDNNSetBiasDesc<T>(B_desc_, X_dims.size(), W_dims[0], data_format);
    }
    workspace_size_ = SIZE_MAX;
    workspace_maxsize_ = CUDNN_CONV_WORKSPACE_LIMIT_BYTES;
    if (CUDAContext::objects().cudnn_benchmark_) {
      exhaustive_search_ = true;
      return;
    }
    exhaustive_search_ = false;
    if (CUDAContext::objects().cudnn_deterministic_) {
      algo_ = ConvAlgoSearch<AlgoT>().get_deterministic();
      return;
    }
    auto fn = [&]() {
      return AlgoWithCost(
          ConvAlgoSearch<AlgoT>().get(
              ctx->cudnn_handle(),
              X_desc_,
              Y_desc_,
              conv_desc_,
              W_desc_,
              workspace_maxsize_),
          0.f);
    };
    auto result = algo_cache_.get(X_dims, W_dims, TypeMeta::Id<T>(), fn);
    algo_ = std::get<0>(result);
  }

  template <typename T>
  void Compute(const T* dY, const T* X, T* dW, T* dB, CUDAContext* ctx) {
    if (exhaustive_search_) {
      exhaustive_search_ = false;
      auto fn = [&]() {
        return ConvAlgoSearch<AlgoT>().find(
            ctx->cudnn_handle(),
            X_desc_,
            Y_desc_,
            conv_desc_,
            W_desc_,
            workspace_maxsize_,
            X,
            dY,
            dW,
            ctx->workspace()->data<CUDAContext>(workspace_maxsize_));
      };
      auto result = algo_cache_.get(X_dims_, W_dims_, TypeMeta::Id<T>(), fn);
      algo_ = std::get<0>(result);
    }
    if (workspace_size_ == SIZE_MAX) {
      auto algo_status = CUDNN_STATUS_SUCCESS;
      for (int step = 0; step < 2; ++step) {
        algo_status = cudnnGetConvolutionBackwardFilterWorkspaceSize(
            ctx->cudnn_handle(),
            X_desc_,
            Y_desc_,
            conv_desc_,
            W_desc_,
            algo_,
            &workspace_size_);
        if (algo_status == CUDNN_STATUS_SUCCESS) break;
        algo_ = ConvAlgoSearch<AlgoT>().get_default();
      }
      CUDNN_CHECK(algo_status);
    }
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(
        ctx->cudnn_handle(),
        CuDNNTraits<T>::one,
        X_desc_,
        X,
        Y_desc_,
        dY,
        conv_desc_,
        algo_,
        ctx->workspace()->data<CUDAContext>(workspace_size_),
        workspace_size_,
        CuDNNTraits<T>::zero,
        W_desc_,
        dW));
    if (dB != nullptr && TypeMeta::Id<T>() != TypeMeta::Id<bfloat16>()) {
      CUDNN_CHECK(cudnnConvolutionBackwardBias(
          ctx->cudnn_handle(),
          CuDNNTraits<T>::one,
          Y_desc_,
          dY,
          CuDNNTraits<T>::zero,
          B_desc_,
          dB));
    }
  }

  vec64_t X_dims_, W_dims_;
  size_t workspace_size_, workspace_maxsize_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnFilterDescriptor_t W_desc_;
  cudnnTensorDescriptor_t X_desc_, Y_desc_, B_desc_;
  AlgoT algo_;
  ConvAlgoCache<AlgoWithCost> algo_cache_;
  bool exhaustive_search_;
};

} // namespace dragon

#endif // USE_CUDNN

#endif // DRAGON_OPERATORS_VISION_CONV_OP_IMPL_CUDNN_H_
