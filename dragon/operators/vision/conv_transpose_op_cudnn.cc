#ifdef USE_CUDNN

#include "dragon/core/workspace.h"
#include "dragon/operators/vision/conv_transpose_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CuDNNConvTransposeOp<Context>::SetOpDesc() {
  auto &X = Input(0), &W = Input(1), *Y = Output(0);
  bool input_changed = (X.dims() != input_dims_);
  bool filter_changed = (W.dims() != filter_dims_);
  if (!input_changed && !filter_changed) return;
  if (input_changed) {
    input_dims_ = X.dims();
    CuDNNSetTensorDesc<T>(&input_desc_, X.dims(), data_format());
    CuDNNSetTensorDesc<T>(&output_desc_, Y->dims(), data_format());
  }
  if (filter_changed) {
    filter_dims_ = W.dims();
    this->template SetFilterDesc<T>();
    CuDNNSetBiasDesc<T>(&bias_desc_, X.ndim(), out_channels_, data_format());
  }
  this->template SetConvDesc<T>();
  scratch_size_ = SIZE_MAX;
  scratch_max_size_ = CUDNN_CONV_WORKSPACE_LIMIT_BYTES;
  if (CUDAContext::objects().cudnn_benchmark_) {
    exhaustive_search_ = true;
    return;
  }
  exhaustive_search_ = false;
  if (CUDAContext::objects().cudnn_deterministic_) {
    fwd_algo_ = ConvAlgoSearch<FwdAlgo>().get_deterministic();
    return;
  }
  auto fn = [&]() {
    return std::tuple<FwdAlgo, float>(
        ConvAlgoSearch<FwdAlgo>().get(
            ctx()->cudnn_handle(),
            filter_desc_,
            input_desc_,
            conv_desc_,
            output_desc_,
            scratch_max_size_),
        0.f);
  };
  auto result = fwd_algo_cache_.get(X.dims(), W.dims(), compute_type_, fn);
  fwd_algo_ = std::get<0>(result);
}

template <class Context>
template <typename T>
void CuDNNConvTransposeOp<Context>::DoRunWithType() {
  ConvOpBase<Context>::Reshape();
  auto &X = Input(0), &W = Input(1), *Y = Output(0);
  INITIALIZE_TENSOR_VIA_SPEC(W, w_shape_, T);
  if (HasBias()) {
    INITIALIZE_TENSOR_VIA_SPEC(Input(2), b_shape_, T);
  }
  SetOpDesc<T>();

  // Find algorithm if required.
  if (exhaustive_search_) {
    exhaustive_search_ = false;
    auto fn = [&]() {
      return ConvAlgoSearch<FwdAlgo>().find(
          ctx()->cudnn_handle(),
          filter_desc_,
          input_desc_,
          conv_desc_,
          output_desc_,
          scratch_max_size_,
          W.template data<T, Context>(),
          X.template data<T, Context>(),
          Y->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(scratch_max_size_));
    };
    auto result = fwd_algo_cache_.get(X.dims(), W.dims(), compute_type_, fn);
    fwd_algo_ = std::get<0>(result);
  }

  // Set workspace for the selected algorithm.
  if (scratch_size_ == SIZE_MAX) {
    auto algo_status = CUDNN_STATUS_SUCCESS;
    for (int step = 0; step < 2; ++step) {
      algo_status = cudnnGetConvolutionBackwardDataWorkspaceSize(
          ctx()->cudnn_handle(),
          filter_desc_,
          input_desc_,
          conv_desc_,
          output_desc_,
          fwd_algo_,
          &scratch_size_);
      if (algo_status == CUDNN_STATUS_SUCCESS) break;
      fwd_algo_ = ConvAlgoSearch<FwdAlgo>().get_default();
    }
    CUDNN_CHECK(algo_status);
  }

  CUDNN_CHECK(cudnnConvolutionBackwardData(
      ctx()->cudnn_handle(),
      CuDNNType<T>::one,
      filter_desc_,
      W.template data<T, Context>(),
      input_desc_,
      X.template data<T, Context>(),
      conv_desc_,
      fwd_algo_,
      ctx()->workspace()->template data<Context>(scratch_size_),
      scratch_size_,
      CuDNNType<T>::zero,
      output_desc_,
      Y->template mutable_data<T, Context>()));

  if (HasBias()) {
    CUDNN_CHECK(cudnnAddTensor(
        ctx()->cudnn_handle(),
        CuDNNType<T>::one,
        bias_desc_,
        Input(2).template data<T, Context>(),
        CuDNNType<T>::one,
        output_desc_,
        Y->template mutable_data<T, Context>()));
  }
}

template <class Context>
template <typename T>
void CuDNNConvTransposeGradientOp<Context>::SetOpDesc() {
  auto &X = Input(0), &W = Input(1), &dY = Input(-1);
  bool input_changed = (X.dims() != input_dims_);
  bool filter_changed = (W.dims() != filter_dims_);
  if (!input_changed && !filter_changed) return;
  if (input_changed) {
    input_dims_ = X.dims();
    CuDNNSetTensorDesc<T>(&input_desc_, dY.dims(), data_format());
    CuDNNSetTensorDesc<T>(&output_desc_, X.dims(), data_format());
  }
  if (filter_changed) {
    filter_dims_ = W.dims();
    this->template SetFilterDesc<T>();
    CuDNNSetBiasDesc<T>(&bias_desc_, X.ndim(), out_channels_, data_format());
  }
  this->template SetConvDesc<T>();
  scratch_size_ = SIZE_MAX;
  scratch_max_size_ = CUDNN_CONV_WORKSPACE_LIMIT_BYTES;
  if (CUDAContext::objects().cudnn_benchmark_) {
    exhaustive_search_data_ = exhaustive_search_filter_ = true;
    return;
  }
  exhaustive_search_data_ = exhaustive_search_filter_ = false;
  if (CUDAContext::objects().cudnn_deterministic_) {
    bwd_data_algo_ = ConvAlgoSearch<BwdDataAlgo>().get_deterministic();
    bwd_filter_algo_ = ConvAlgoSearch<BwdFilterAlgo>().get_deterministic();
    return;
  }
  {
    auto fn = [&]() {
      return std::tuple<BwdDataAlgo, float>(
          ConvAlgoSearch<BwdDataAlgo>().get(
              ctx()->cudnn_handle(),
              input_desc_,
              filter_desc_,
              conv_desc_,
              output_desc_,
              scratch_max_size_),
          0.f);
    };
    auto result = data_algo_cache_.get(X.dims(), W.dims(), compute_type_, fn);
    bwd_data_algo_ = std::get<0>(result);
  }
  {
    auto fn = [&]() {
      return std::tuple<BwdFilterAlgo, float>(
          ConvAlgoSearch<BwdFilterAlgo>().get(
              ctx()->cudnn_handle(),
              input_desc_,
              output_desc_,
              conv_desc_,
              filter_desc_,
              scratch_max_size_),
          0.f);
    };
    auto result = filter_algo_cache_.get(X.dims(), W.dims(), compute_type_, fn);
    bwd_filter_algo_ = std::get<0>(result);
  }
}

template <class Context>
template <typename T>
void CuDNNConvTransposeGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &W = Input(1), &dY = Input(-1);
  auto *dX = Output(0), *dW = Output(1);
  SetOpDesc<T>();

  // Find algorithm if required.
  if (dW->has_name() && exhaustive_search_filter_) {
    exhaustive_search_filter_ = false;
    auto fn = [&]() {
      return ConvAlgoSearch<BwdFilterAlgo>().find(
          ctx()->cudnn_handle(),
          input_desc_,
          output_desc_,
          conv_desc_,
          filter_desc_,
          scratch_max_size_,
          dY.template data<T, Context>(),
          X.template data<T, Context>(),
          dW->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(scratch_max_size_));
    };
    auto result = filter_algo_cache_.get(X.dims(), W.dims(), compute_type_, fn);
    bwd_filter_algo_ = std::get<0>(result);
  }
  if (dX->has_name() && exhaustive_search_data_) {
    exhaustive_search_data_ = false;
    auto fn = [&]() {
      return ConvAlgoSearch<BwdDataAlgo>().find(
          ctx()->cudnn_handle(),
          input_desc_,
          filter_desc_,
          conv_desc_,
          output_desc_,
          scratch_max_size_,
          dY.template data<T, Context>(),
          W.template data<T, Context>(),
          dX->template mutable_data<T, Context>(),
          ctx()->workspace()->template data<Context>(scratch_max_size_));
    };
    auto result = data_algo_cache_.get(X.dims(), W.dims(), compute_type_, fn);
    bwd_data_algo_ = std::get<0>(result);
  }

  // Set workspace for the selected algorithm.
  if (scratch_size_ == SIZE_MAX) {
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
      if (algo_status == CUDNN_STATUS_SUCCESS) break;
      bwd_filter_algo_ = ConvAlgoSearch<BwdFilterAlgo>().get_default();
    }
    CUDNN_CHECK(algo_status);
    for (int step = 0; step < 2; ++step) {
      algo_status = cudnnGetConvolutionForwardWorkspaceSize(
          ctx()->cudnn_handle(),
          input_desc_,
          filter_desc_,
          conv_desc_,
          output_desc_,
          bwd_data_algo_,
          &bwd_data_size);
      if (algo_status == CUDNN_STATUS_SUCCESS) break;
      bwd_data_algo_ = ConvAlgoSearch<BwdDataAlgo>().get_default();
    }
    CUDNN_CHECK(algo_status);
    scratch_size_ = std::max(bwd_filter_size, bwd_data_size);
  }

  if (Output(2)->has_name()) {
    CUDNN_CHECK(cudnnConvolutionBackwardBias(
        ctx()->cudnn_handle(),
        CuDNNType<T>::one,
        input_desc_,
        dY.template data<T, Context>(),
        CuDNNType<T>::zero,
        bias_desc_,
        Output(2)->template mutable_data<T, Context>()));
  }

  if (dW->has_name()) {
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(
        ctx()->cudnn_handle(),
        CuDNNType<T>::one,
        input_desc_,
        dY.template data<T, Context>(),
        output_desc_,
        X.template data<T, Context>(),
        conv_desc_,
        bwd_filter_algo_,
        ctx()->workspace()->template data<Context>(scratch_size_),
        scratch_size_,
        CuDNNType<T>::zero,
        filter_desc_,
        dW->template mutable_data<T, Context>()));
  }

  if (dX->has_name()) {
    CUDNN_CHECK(cudnnConvolutionForward(
        ctx()->cudnn_handle(),
        CuDNNType<T>::one,
        input_desc_,
        dY.template data<T, Context>(),
        filter_desc_,
        W.template data<T, Context>(),
        conv_desc_,
        bwd_data_algo_,
        ctx()->workspace()->template data<Context>(scratch_size_),
        scratch_size_,
        CuDNNType<T>::zero,
        output_desc_,
        dX->template mutable_data<T, Context>()));
  }
}

template <class Context>
void CuDNNConvTransposeGradientOp<Context>::RunOnDevice() {
  ConvOpBase<Context>::Reshape(true);
  DispatchHelper<dtypes::Floating>::Call(this, Input(-1));
}

DEPLOY_CUDNN_OPERATOR(ConvTranspose);
DEPLOY_CUDNN_OPERATOR(ConvTransposeGradient);

} // namespace dragon

#endif // USE_CUDNN
