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

#ifndef DRAGON_OPERATORS_MATH_REDUCE_OP_IMPL_CNNL_H_
#define DRAGON_OPERATORS_MATH_REDUCE_OP_IMPL_CNNL_H_

#ifdef USE_MLU

#include "dragon/core/context_mlu.h"
#include "dragon/utils/math/reduce.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

class CNNLReduceOpImpl {
 public:
  CNNLReduceOpImpl() : scratch_size_(0) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&output_desc_);
    index_mode_ = CNNL_REDUCE_NO_INDICES;
    CNNL_CHECK(cnnlCreateReduceDescriptor(&reduce_desc_));
  }

  ~CNNLReduceOpImpl() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(output_desc_);
    CNNL_CHECK(cnnlDestroyReduceDescriptor(reduce_desc_));
  }

  void SetReducer(const cnnlReduceOp_t reducer) {
    reducer_ = reducer;
  }

  void SetIndexMode(const cnnlReduceIndices_t index_mode) {
    index_mode_ = index_mode;
  }

  template <typename T>
  void Setup(const vec64_t& X_dims, const vec64_t& X_axes, MLUContext* ctx) {
    Setup<T>(reducer_, X_dims, X_axes, ctx);
  }

  template <typename T>
  void Setup(
      const cnnlReduceOp_t reducer,
      const vec64_t& X_dims,
      const vec64_t& X_axes,
      MLUContext* ctx) {
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
    CNNLSetTensorDesc<T>(input_desc_, input_dims);
    CNNLSetTensorDesc<T>(output_desc_, output_dims);
    cnnlDataType_t reduce_type = CNNLGetDataType<T>();
    if (reduce_type == CNNL_DTYPE_HALF || reduce_type == CNNL_DTYPE_BFLOAT16) {
      reduce_type = CNNL_DTYPE_FLOAT;
    }
    if (reducer == CNNL_REDUCE_MAX || reducer == CNNL_REDUCE_MIN) {
      reduce_type = CNNLGetDataType<T>();
    }
    CNNL_CHECK(cnnlSetReduceDescriptor_v2(
        reduce_desc_,
        vec32_t({reduce_axes.begin(), reduce_axes.end()}).data(),
        reduce_axes.size(),
        reducer,
        reduce_type,
        CNNL_NOT_PROPAGATE_NAN,
        index_mode_,
        CNNL_32BIT_INDICES,
        0.f));
    CNNL_CHECK(cnnlGetReduceOpWorkspaceSize(
        ctx->cnnl_handle(),
        input_desc_,
        output_desc_,
        reduce_desc_,
        &scratch_size_));
  }

  template <typename T>
  void Compute(
      const T* X,
      T* Y,
      void* scratch,
      MLUContext* ctx,
      const float scale = 1.f,
      const float bias = 0.f) {
    const T beta = convert::To<T>(bias);
    const T alpha = convert::To<T>(scale);
    CNNL_CHECK(cnnlReduce(
        ctx->cnnl_handle(),
        reduce_desc_,
        scratch,
        scratch_size_,
        scale != 1.f ? &alpha : nullptr,
        input_desc_,
        X,
        0, // indices bytes
        nullptr, // indices
        bias != 0.f ? &beta : nullptr,
        output_desc_,
        Y));
  }

  template <typename T>
  void ComputeIndex(const T* X, int* Y, void* scratch, MLUContext* ctx) {
    CHECK_EQ(index_mode_, CNNL_REDUCE_ONLY_INDICES);
    CNNL_CHECK(cnnlReduce(
        ctx->cnnl_handle(),
        reduce_desc_,
        scratch,
        scratch_size_,
        nullptr,
        input_desc_,
        X,
        sizeof(int) * cnnlGetTensorElementNum(output_desc_),
        Y, // indices
        nullptr,
        output_desc_,
        nullptr));
  }

  size_t scratch_size() {
    return scratch_size_;
  }

 private:
  size_t scratch_size_;
  cnnlReduceOp_t reducer_;
  cnnlReduceIndices_t index_mode_;
  cnnlReduceDescriptor_t reduce_desc_;
  cnnlTensorDescriptor_t input_desc_, output_desc_;
};

} // namespace dragon

#endif // USE_MLU

#endif // DRAGON_OPERATORS_MATH_REDUCE_OP_IMPL_CNNL_H_
