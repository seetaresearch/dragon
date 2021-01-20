#ifdef USE_CUDNN

#include "dragon/core/workspace.h"
#include "dragon/operators/vision/conv_transpose_op.h"
#include "dragon/utils/filler.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNConvTransposeOp<Context>::ResetDesc() {
  auto &X = Input(0), &W = Input(1), *Y = Output(0);
  bool input_changed = (X.dims() != input_dims_);
  bool filter_changed = (W.dims() != filter_dims_);
  if (input_changed || filter_changed) {
    if (input_changed) {
      input_dims_ = X.dims();
      CuDNNSetTensorDesc<T>(&input_desc_, X.dims(), data_format(), group_v2_);
      CuDNNSetTensorDesc<T>(&output_desc_, Y->dims(), data_format(), group_v2_);
      if (HasBias()) {
        CuDNNSetTensorDesc<T>(&output_desc_for_bias_, Y->dims(), data_format());
      }
    }
    if (filter_changed) {
      filter_dims_ = W.dims();
      this->template SetFilterDesc<T>();
      if (HasBias()) {
        CuDNNSetBiasDesc<T>(
            &bias_desc_, X.ndim(), out_channels_, data_format());
      }
    }
    this->template SetConvDesc<T>();
    // Get or search the appropriate algorithm
    if (CUDAContext::objects().cudnn_deterministic_) {
      fwd_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    } else if (CUDAContext::objects().cudnn_benchmark_) {
      exhaustive_search_ = true;
    } else {
#if CUDNN_VERSION_MIN(7, 0, 0)
      int num_valid_algos;
      constexpr int num_algos = CUDNN_CONV_NUM_BWD_DATA_ALGOS;
      cudnnConvolutionBwdDataAlgoPerf_t stats[num_algos];
      CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
          ctx()->cudnn_handle(),
          filter_desc_,
          input_desc_,
          conv_desc_,
          output_desc_,
          num_algos,
          &num_valid_algos,
          stats));
      bool algo_is_found = false;
      for (int i = 0; i < num_valid_algos; ++i) {
        if (stats[i].memory <= CUDNN_CONV_WORKSPACE_LIMIT_BYTES) {
          fwd_algo_ = stats[i].algo;
          algo_is_found = true;
          break;
        }
      }
      CHECK(algo_is_found)
          << "\nNo algorithms available for <cudnnConvolutionBackwardData> "
          << "under the current desc and workspace limit.";
#else
      CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
          ctx()->cudnn_handle(),
          filter_desc_,
          input_desc_,
          conv_desc_,
          output_desc_,
          CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
          CUDNN_CONV_WORKSPACE_LIMIT_BYTES,
          &fwd_algo_));
#endif // CUDNN_VERSION_MIN(7, 0, 0)
    }
    cudnn_ws_nbytes_ = SIZE_MAX; // Request a new size
  }
}

template <class Context>
template <typename T>
void CuDNNConvTransposeOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1);

  TENSOR_FILL(W, w_shape_);
  if (HasBias()) {
    TENSOR_FILL(Input(2), b_shape_);
  }

  ResetDesc<T>();

  auto* x = X.template data<T, Context>();
  auto* w = W.template data<T, Context>();
  auto* y = Output(0)->template mutable_data<T, Context>();

  void* scratch = nullptr; // workspace buffer

  // Find the appropriate algorithm if necessary
  if (exhaustive_search_) {
    scratch = ctx()->workspace()->template data<Context>(
        {CUDNN_CONV_WORKSPACE_LIMIT_BYTES})[0];
    auto algo = algo_cache_.get(X.dims(), W.dims(), compute_type_, [&]() {
      int num_valid_algos;
      constexpr int num_algos = CUDNN_CONV_NUM_BWD_DATA_ALGOS;
      cudnnConvolutionBwdDataAlgoPerf_t stats[num_algos];
      CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithmEx(
          ctx()->cudnn_handle(),
          filter_desc_,
          w,
          input_desc_,
          x,
          conv_desc_,
          output_desc_,
          y,
          num_algos,
          &num_valid_algos,
          stats,
          scratch,
          CUDNN_CONV_WORKSPACE_LIMIT_BYTES));
      return FwdAlgoWithCost(stats[0].algo, stats[0].time);
    });
    exhaustive_search_ = false;
    fwd_algo_ = std::get<0>(algo);
  }

  // Determine the workspace size for selected algorithm
  if (cudnn_ws_nbytes_ == SIZE_MAX) {
    auto algo_status = CUDNN_STATUS_SUCCESS;
    for (int step = 0; step < 2; ++step) {
      algo_status = cudnnGetConvolutionBackwardDataWorkspaceSize(
          ctx()->cudnn_handle(),
          filter_desc_,
          input_desc_,
          conv_desc_,
          output_desc_,
          fwd_algo_,
          &cudnn_ws_nbytes_);
      if (algo_status != CUDNN_STATUS_SUCCESS && step == 0 &&
          CUDAContext::objects().cudnn_deterministic_) {
        fwd_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
      } else {
        CUDNN_CHECK(algo_status);
      }
    }
  }

  // Alloc the memory for workspace data
  if (cudnn_ws_nbytes_ > 0) {
    scratch = ctx()->workspace()->template data<Context>({cudnn_ws_nbytes_})[0];
  }

  for (int g = 0; g < group_v2_; g++) {
    CUDNN_CHECK(cudnnConvolutionBackwardData(
        ctx()->cudnn_handle(),
        CuDNNType<T>::one,
        filter_desc_,
        w + W_stride_ * g,
        input_desc_,
        x + X_stride_ * g,
        conv_desc_,
        fwd_algo_,
        scratch,
        cudnn_ws_nbytes_,
        CuDNNType<T>::zero,
        output_desc_,
        y + Y_stride_ * g));
  }

  if (HasBias()) {
    auto* b = Input(2).template data<T, Context>();
    CUDNN_CHECK(cudnnAddTensor(
        ctx()->cudnn_handle(),
        CuDNNType<T>::one,
        bias_desc_,
        b,
        CuDNNType<T>::one,
        output_desc_for_bias_,
        y));
  }
}

template <class Context>
void CuDNNConvTransposeOp<Context>::RunOnDevice() {
  ConvOpBase<Context>::Reshape();
  if (data_format() == "NCHW") {
    X_stride_ = Input(0).stride(0) / group_v2_;
    Y_stride_ = Output(0)->stride(0) / group_v2_;
  } else if (data_format() == "NHWC") {
    X_stride_ = Input(0).dim(-1) / group_v2_;
    Y_stride_ = Output(0)->dim(-1) / group_v2_;
  }
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void CuDNNConvTransposeGradientOp<Context>::ResetDesc() {
  auto &X = Input(0), &W = Input(1), &dY = Input(-1);
  bool input_changed = (X.dims() != input_dims_);
  bool filter_changed = (W.dims() != filter_dims_);
  if (input_changed || filter_changed) {
    if (input_changed) {
      input_dims_ = X.dims();
      CuDNNSetTensorDesc<T>(&input_desc_, dY.dims(), data_format(), group_v2_);
      CuDNNSetTensorDesc<T>(&output_desc_, X.dims(), data_format(), group_v2_);
      if (HasBias()) {
        CuDNNSetTensorDesc<T>(&input_desc_for_bias_, dY.dims(), data_format());
      }
    }
    if (filter_changed) {
      filter_dims_ = W.dims();
      this->template SetFilterDesc<T>();
      if (HasBias()) {
        CuDNNSetBiasDesc<T>(
            &bias_desc_, X.ndim(), out_channels_, data_format());
      }
    }
    this->template SetConvDesc<T>();
    // Get the appropriate algorithm
    if (CUDAContext::objects().cudnn_deterministic_) {
      bwd_data_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
      bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    } else if (CUDAContext::objects().cudnn_benchmark_) {
      exhaustive_search_data_ = true;
      exhaustive_search_filter_ = true;
    } else {
#if CUDNN_VERSION_MIN(7, 0, 0)
      {
        int num_valid_algos;
        constexpr int num_algos = CUDNN_CONV_NUM_BWD_FILTER_ALGOS;
        cudnnConvolutionBwdFilterAlgoPerf_t stats[num_algos];
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            ctx()->cudnn_handle(),
            input_desc_,
            output_desc_,
            conv_desc_,
            filter_desc_,
            num_algos,
            &num_valid_algos,
            stats));
        bool algo_is_found = false;
        for (int i = 0; i < num_valid_algos; ++i) {
          if (stats[i].memory <= CUDNN_CONV_WORKSPACE_LIMIT_BYTES) {
            bwd_filter_algo_ = stats[i].algo;
            algo_is_found = true;
            break;
          }
        }
        CHECK(algo_is_found)
            << "\nNo algorithms available for <cudnnConvolutionBackwardFilter> "
            << "under the current desc and workspace limit.";
      }
      {
        int num_valid_algos;
        constexpr int num_algos = CUDNN_CONV_NUM_FWD_ALGOS;
        cudnnConvolutionFwdAlgoPerf_t stats[num_algos];
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
            ctx()->cudnn_handle(),
            input_desc_,
            filter_desc_,
            conv_desc_,
            output_desc_,
            num_algos,
            &num_valid_algos,
            stats));
        bool algo_is_found = false;
        for (int i = 0; i < num_valid_algos; ++i) {
          if (stats[i].memory <= CUDNN_CONV_WORKSPACE_LIMIT_BYTES) {
            bwd_data_algo_ = stats[i].algo;
            algo_is_found = true;
            break;
          }
        }
        CHECK(algo_is_found)
            << "\nNo algorithms available for <cudnnConvolutionForward> "
            << "under the current desc and workspace limit.";
      }
#else
      CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
          ctx()->cudnn_handle(),
          input_desc_,
          output_desc_,
          conv_desc_,
          filter_desc_,
          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
          CUDNN_CONV_WORKSPACE_LIMIT_BYTES,
          &bwd_filter_algo_));
      CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
          ctx()->cudnn_handle(),
          input_desc_,
          filter_desc_,
          conv_desc_,
          output_desc_,
          CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
          CUDNN_CONV_WORKSPACE_LIMIT_BYTES,
          &bwd_data_algo_));
#endif // CUDNN_VERSION_MIN(7, 0, 0)
    }
    cudnn_ws_nbytes_ = SIZE_MAX; // Request a new size
  }
}

template <class Context>
template <typename T>
void CuDNNConvTransposeGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1);
  auto *dX = Output(0), *dW = Output(1);
  ResetDesc<T>();

  const T *x = nullptr, *w = nullptr;
  T *dx = nullptr, *dw = nullptr;
  void* scratch = nullptr; // workspace buffer
  auto* dy = Input(-1).template data<T, Context>();

  // Find the appropriate algorithm if necessary
  if (dW->has_name() && exhaustive_search_filter_) {
    scratch = ctx()->workspace()->template data<Context>(
        {CUDNN_CONV_WORKSPACE_LIMIT_BYTES})[0];
    x = X.template data<T, Context>();
    dw = dW->template mutable_data<T, Context>();
    auto algo =
        filter_algo_cache_.get(X.dims(), W.dims(), compute_type_, [&]() {
          int num_valid_algos;
          constexpr int num_algos = CUDNN_CONV_NUM_BWD_FILTER_ALGOS;
          cudnnConvolutionBwdFilterAlgoPerf_t stats[num_algos];
          CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithmEx(
              ctx()->cudnn_handle(),
              input_desc_,
              dy,
              output_desc_,
              x,
              conv_desc_,
              filter_desc_,
              dw,
              num_algos,
              &num_valid_algos,
              stats,
              scratch,
              CUDNN_CONV_WORKSPACE_LIMIT_BYTES));
          return BwdFilterAlgoWithCost(stats[0].algo, stats[0].time);
        });
    exhaustive_search_filter_ = false;
    bwd_filter_algo_ = std::get<0>(algo);
  }

  if (dX->has_name() && exhaustive_search_data_) {
    scratch = ctx()->workspace()->template data<Context>(
        {CUDNN_CONV_WORKSPACE_LIMIT_BYTES})[0];
    w = W.template data<T, Context>();
    dx = dX->template mutable_data<T, Context>();
    auto algo = data_algo_cache_.get(X.dims(), W.dims(), compute_type_, [&]() {
      int num_valid_algos;
      constexpr int num_algos = CUDNN_CONV_NUM_FWD_ALGOS;
      cudnnConvolutionFwdAlgoPerf_t stats[num_algos];
      CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(
          ctx()->cudnn_handle(),
          input_desc_,
          dy,
          filter_desc_,
          w,
          conv_desc_,
          output_desc_,
          dx,
          num_algos,
          &num_valid_algos,
          stats,
          scratch,
          CUDNN_CONV_WORKSPACE_LIMIT_BYTES));
      return BwdDataAlgoWithCost(stats[0].algo, stats[0].time);
    });
    exhaustive_search_data_ = false;
    bwd_data_algo_ = std::get<0>(algo);
  }

  // Determine the workspace size for selected algorithm
  if (cudnn_ws_nbytes_ == SIZE_MAX) {
    auto algo_status = CUDNN_STATUS_SUCCESS;
    size_t bwd_filter_size = 0, bwd_data_size = 0;
    for (int step = 0; step < 2; ++step) {
      algo_status = cudnnGetConvolutionBackwardFilterWorkspaceSize(
          ctx()->cudnn_handle(),
          input_desc_,
          output_desc_,
          conv_desc_,
          filter_desc_,
          bwd_filter_algo_,
          &bwd_filter_size);
      if (algo_status != CUDNN_STATUS_SUCCESS && step == 0 &&
          CUDAContext::objects().cudnn_deterministic_) {
        bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
      } else {
        CUDNN_CHECK(algo_status);
      }
    }
    for (int step = 0; step < 2; ++step) {
      algo_status = cudnnGetConvolutionForwardWorkspaceSize(
          ctx()->cudnn_handle(),
          input_desc_,
          filter_desc_,
          conv_desc_,
          output_desc_,
          bwd_data_algo_,
          &bwd_data_size);
      if (algo_status != CUDNN_STATUS_SUCCESS && step == 0 &&
          CUDAContext::objects().cudnn_deterministic_) {
        bwd_data_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
      } else {
        CUDNN_CHECK(algo_status);
      }
    }
    cudnn_ws_nbytes_ = std::max(bwd_filter_size, bwd_data_size);
  }

  // Alloc the memory for workspace data
  if (cudnn_ws_nbytes_ > 0) {
    scratch = ctx()->workspace()->template data<Context>({cudnn_ws_nbytes_})[0];
  }

  if (Output(2)->has_name()) {
    auto* db = Output(2)->template mutable_data<T, Context>();
    CUDNN_CHECK(cudnnConvolutionBackwardBias(
        ctx()->cudnn_handle(),
        CuDNNType<T>::one,
        input_desc_for_bias_,
        dy,
        CuDNNType<T>::zero,
        bias_desc_,
        db));
  }

  if (dW->has_name()) {
    x = X.template data<T, Context>();
    dw = dW->template mutable_data<T, Context>();
    for (int g = 0; g < group_v2_; g++) {
      CUDNN_CHECK(cudnnConvolutionBackwardFilter(
          ctx()->cudnn_handle(),
          CuDNNType<T>::one,
          input_desc_,
          dy + Y_stride_ * g,
          output_desc_,
          x + X_stride_ * g,
          conv_desc_,
          bwd_filter_algo_,
          scratch,
          cudnn_ws_nbytes_,
          CuDNNType<T>::zero,
          filter_desc_,
          dw + W_stride_ * g));
    }
  }

  if (dX->has_name()) {
    auto* w = W.template data<T, Context>();
    auto* dx = dX->template mutable_data<T, Context>();
    for (int g = 0; g < group_v2_; g++) {
      CUDNN_CHECK(cudnnConvolutionForward(
          ctx()->cudnn_handle(),
          CuDNNType<T>::one,
          input_desc_,
          dy + Y_stride_ * g,
          filter_desc_,
          w + W_stride_ * g,
          conv_desc_,
          bwd_data_algo_,
          scratch,
          cudnn_ws_nbytes_,
          CuDNNType<T>::zero,
          output_desc_,
          dx + X_stride_ * g));
    }
  }
}

template <class Context>
void CuDNNConvTransposeGradientOp<Context>::RunOnDevice() {
  ConvOpBase<Context>::Reshape(true);
  if (data_format() == "NCHW") {
    X_stride_ = Input(0).stride(0) / group_v2_;
    Y_stride_ = Input(-1).stride(0) / group_v2_;
  } else if (data_format() == "NHWC") {
    X_stride_ = Input(0).dim(-1) / group_v2_;
    Y_stride_ = Input(-1).dim(-1) / group_v2_;
  }
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(-1));
}

DEPLOY_CUDNN_OPERATOR(ConvTranspose);
DEPLOY_CUDNN_OPERATOR(ConvTransposeGradient);

} // namespace dragon

#endif // USE_CUDNN
