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

#ifndef DRAGON_OPERATORS_MATH_REDUCE_OP_IMPL_CUDNN_H_
#define DRAGON_OPERATORS_MATH_REDUCE_OP_IMPL_CUDNN_H_

#ifdef USE_CUDNN

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math/reduce.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

class CuDNNReduceOpImpl {
 public:
  CuDNNReduceOpImpl() : scratch_size_(0) {
    CuDNNCreateTensorDesc(&input_desc_);
    CuDNNCreateTensorDesc(&output_desc_);
    index_mode_ = CUDNN_REDUCE_TENSOR_NO_INDICES;
    CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&reduce_desc_));
  }

  ~CuDNNReduceOpImpl() {
    CuDNNDestroyTensorDesc(input_desc_);
    CuDNNDestroyTensorDesc(output_desc_);
    CUDNN_CHECK(cudnnDestroyReduceTensorDescriptor(reduce_desc_));
  }

  void SetReducer(const cudnnReduceTensorOp_t reducer) {
    reducer_ = reducer;
  }

  void SetIndexMode(const cudnnReduceTensorIndices_t index_mode) {
    index_mode_ = index_mode;
  }

  template <typename T>
  void Setup(const vec64_t& X_dims, const vec64_t& X_axes, CUDAContext* ctx) {
    Setup<T>(reducer_, X_dims, X_axes, ctx);
  }

  template <typename T>
  void Setup(
      const cudnnReduceTensorOp_t reducer,
      const vec64_t& X_dims,
      const vec64_t& X_axes,
      CUDAContext* ctx) {
    vec64_t input_dims, reduce_axes, output_dims;
    math::utils::CollapseReduceAxes(
        X_dims.size(),
        X_dims.data(),
        X_axes.size(),
        X_axes.data(),
        input_dims,
        reduce_axes);
    output_dims = input_dims;
    for (const auto axis : reduce_axes) {
      output_dims[axis] = 1;
    }
    CuDNNSetTensorDesc<T>(input_desc_, input_dims);
    CuDNNSetTensorDesc<T>(output_desc_, output_dims);
    reduce_type_ = CuDNNGetDataType<T>();
    if (reduce_type_ == CUDNN_DATA_HALF ||
        reduce_type_ == CUDNN_DATA_BFLOAT16) {
      reduce_type_ = CUDNN_DATA_FLOAT;
    }
    if (reducer == CUDNN_REDUCE_TENSOR_MIN ||
        reducer == CUDNN_REDUCE_TENSOR_MAX ||
        reducer == CUDNN_REDUCE_TENSOR_AMAX) {
      reduce_type_ = CuDNNGetDataType<T>();
    }
    CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
        reduce_desc_,
        reducer,
        reduce_type_,
        CUDNN_NOT_PROPAGATE_NAN,
        index_mode_,
        CUDNN_32BIT_INDICES));
    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
        ctx->cudnn_handle(),
        reduce_desc_,
        input_desc_,
        output_desc_,
        &scratch_size_));
  }

  template <typename T>
  void Compute(
      const T* X,
      T* Y,
      void* scratch,
      CUDAContext* ctx,
      const float scale = 1.f,
      const float bias = 0.f) {
    const T beta = convert::To<T>(bias);
    const T alpha = convert::To<T>(scale);
    CUDNN_CHECK(cudnnReduceTensor(
        ctx->cudnn_handle(),
        reduce_desc_,
        nullptr, // indices
        0, // indices bytes
        scratch,
        scratch_size_,
        reduce_type_ == CUDNN_DATA_FLOAT ? (T*)&scale : &alpha,
        input_desc_,
        X,
        reduce_type_ == CUDNN_DATA_FLOAT ? (T*)&bias : &beta,
        output_desc_,
        Y));
  }

  size_t scratch_size() {
    return scratch_size_;
  }

 private:
  size_t scratch_size_;
  cudnnReduceTensorOp_t reducer_;
  cudnnReduceTensorIndices_t index_mode_;
  cudnnReduceTensorDescriptor_t reduce_desc_;
  cudnnDataType_t reduce_type_;
  cudnnTensorDescriptor_t input_desc_, output_desc_;
};

} // namespace dragon

#endif // USE_CUDNN

#endif // DRAGON_OPERATORS_MATH_REDUCE_OP_IMPL_CUDNN_H_
