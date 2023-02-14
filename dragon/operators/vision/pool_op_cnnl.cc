#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/vision/pool_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLPoolOp<Context>::DoRunWithType() {
  ComputeOutShape();
  auto &X = Input(0), *Y = Output(0);

  this->SetPoolDesc();
  size_t scratch_size = 0, buffer_size = 0;
  CNNL_CHECK(cnnlGetPoolingWorkspaceSize(
      ctx()->cnnl_handle(),
      this->pool_mode_,
      num_axes_ == 1 ? 1 : out_dims_[3],
      out_dims_[2],
      &scratch_size));
  CNNL_CHECK(cnnlGetPoolingExtraInputSize(
      ctx()->cnnl_handle(),
      this->pool_mode_,
      num_axes_ == 1 ? 1 : out_dims_[3],
      out_dims_[2],
      &buffer_size));

  auto* x = X.template data<T, Context>();
  auto* y = Y->Reshape(out_shape_)->template mutable_data<T, Context>();
  float *x_acc = nullptr, *y_acc = nullptr;
  if (TypeMeta::Id<T>() != TypeMeta::Id<float>() && mode_ == "AVG") {
    CNNLSetTensorDesc<float>(this->input_desc_, X.dims(), data_format());
    CNNLSetTensorDesc<float>(this->output_desc_, out_shape_, data_format());
    const auto scratch_count = X.count() + Y->count();
    x_acc = ctx()->workspace()->template data<float, Context>(scratch_count);
    y_acc = x_acc + X.count(); // Temporarily compute at FP32 to avoid FP16Acc.
  } else {
    CNNLSetTensorDesc<T>(this->input_desc_, X.dims(), data_format());
    CNNLSetTensorDesc<T>(this->output_desc_, out_shape_, data_format());
  }

  auto* X_extra = Output("X_extra");
  if (buffer_size > 0 && X_extra->size() != buffer_size) {
    X_extra->Reshape({int64_t(buffer_size)});
    CNNL_CHECK(cnnlInitPoolingExtraInput(
        ctx()->cnnl_handle(),
        this->pool_desc_,
        this->input_desc_,
        this->output_desc_,
        X_extra->template mutable_data<uint8_t, CPUContext>()));
  }

  if (x_acc != nullptr) math::Cast(X.count(), x, x_acc, ctx());
  CNNL_CHECK(cnnlPoolingForward_v2(
      ctx()->cnnl_handle(),
      this->pool_desc_,
      nullptr, // alpha
      this->input_desc_,
      x_acc != nullptr ? x_acc : reinterpret_cast<const float*>(x),
      nullptr, // beta
      buffer_size > 0 ? X_extra->template data<uint8_t, Context>() : nullptr,
      this->output_desc_,
      y_acc != nullptr ? y_acc : reinterpret_cast<float*>(y),
      ctx()->workspace()->template data<Context>(scratch_size, "BufferKernel"),
      scratch_size));
  if (y_acc != nullptr) math::Cast(Y->count(), y_acc, y, ctx());
}

template <class Context>
template <typename T>
void CNNLPoolGradientOp<Context>::DoRunWithType() {
  ComputeOutShape();
  auto &X = Input(0), *dX = Output(0);
  auto &Y = Input(1), &dY = Input(2);

  this->SetPoolDesc();
  CNNLSetTensorDesc<T>(this->input_desc_, dY.dims(), data_format());
  CNNLSetTensorDesc<T>(this->output_desc_, X.dims(), data_format());

  void* max_index;
  if (mode_ == "MAX") {
    if (TypeMeta::Id<T>() == TypeMeta::Id<float16>()) {
      CNNLSetTensorDesc<short>(this->index_desc_, dY.dims(), data_format());
      max_index = ctx()->workspace()->template data<short, Context>(X.count());
    } else {
      CNNLSetTensorDesc<int>(this->index_desc_, dY.dims(), data_format());
      max_index = ctx()->workspace()->template data<int, Context>(X.count());
    }
    CNNL_CHECK(cnnlPoolingIndex(
        ctx()->cnnl_handle(),
        this->pool_desc_,
        this->output_desc_,
        X.template data<T, Context>(),
        this->index_desc_,
        max_index));
  }
  CNNL_CHECK(cnnlPoolingBackward(
      ctx()->cnnl_handle(),
      this->pool_desc_,
      nullptr, // alpha
      mode_ == "MAX" ? this->index_desc_ : this->input_desc_,
      mode_ == "MAX" ? (const T*)max_index : Y.template data<T, Context>(),
      this->input_desc_,
      dY.template data<T, Context>(),
      this->output_desc_,
      X.template data<T, Context>(),
      nullptr, // beta
      this->output_desc_,
      dX->ReshapeLike(X)->template mutable_data<T, Context>()));
}

DEPLOY_CNNL_OPERATOR(Pool);
DEPLOY_CNNL_OPERATOR(PoolGradient);

} // namespace dragon

#endif // USE_MLU
