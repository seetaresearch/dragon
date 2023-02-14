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

#ifndef DRAGON_OPERATORS_MATH_TRANSPOSE_OP_IMPL_CNNL_H_
#define DRAGON_OPERATORS_MATH_TRANSPOSE_OP_IMPL_CNNL_H_

#ifdef USE_MLU

#include "dragon/core/context_mlu.h"
#include "dragon/utils/math/transpose.h"

namespace dragon {

class CNNLTransposeOpImpl {
 public:
  CNNLTransposeOpImpl() : scratch_size_(0) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&output_desc_);
    CNNL_CHECK(cnnlCreateTransposeDescriptor(&transpose_desc_));
  }

  ~CNNLTransposeOpImpl() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(output_desc_);
    CNNL_CHECK(cnnlDestroyTransposeDescriptor(transpose_desc_));
  }

  template <typename T>
  void Setup(const vec64_t& X_dims, const vec64_t& Y_axes, MLUContext* ctx) {
    vec64_t input_dims, transpose_axes, output_dims;
    math::utils::CollapseTransposeAxes(
        X_dims.size(),
        X_dims.data(),
        Y_axes.data(),
        input_dims,
        transpose_axes);
    output_dims.resize(input_dims.size());
    for (int i = 0; i < input_dims.size(); ++i) {
      output_dims[i] = input_dims[transpose_axes[i]];
    }
    CNNLSetTensorDesc<T>(input_desc_, input_dims);
    CNNLSetTensorDesc<T>(output_desc_, output_dims);
    CNNL_CHECK(cnnlSetTransposeDescriptor(
        transpose_desc_,
        input_dims.size(),
        vec32_t({transpose_axes.begin(), transpose_axes.end()}).data()));
    CNNL_CHECK(cnnlGetTransposeWorkspaceSize(
        ctx->cnnl_handle(), input_desc_, transpose_desc_, &scratch_size_));
  }

  template <typename T>
  void Compute(const T* X, T* Y, void* scratch, MLUContext* ctx) {
    CNNL_CHECK(cnnlTranspose_v2(
        ctx->cnnl_handle(),
        transpose_desc_,
        input_desc_,
        X,
        output_desc_,
        Y,
        scratch,
        scratch_size_));
  }

  size_t scratch_size() {
    return scratch_size_;
  }

  size_t scratch_size_;
  cnnlTransposeDescriptor_t transpose_desc_;
  cnnlTensorDescriptor_t input_desc_, output_desc_;
};

} // namespace dragon

#endif // USE_MLU

#endif // DRAGON_OPERATORS_MATH_TRANSPOSE_OP_IMPL_CNNL_H_
