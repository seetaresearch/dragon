#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/vision/pool_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLPoolOp<Context>::DoRunWithType() {
  ComputeOutShape();
  auto &X = Input(0), *Y = Output(0);

  this->SetPoolDesc();
  CNNLSetTensorDesc<T>(this->input_desc_, X.dims(), data_format());
  CNNLSetTensorDesc<T>(this->output_desc_, out_shape_, data_format());

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

  CNNL_CHECK(cnnlPoolingForward_v2(
      ctx()->cnnl_handle(),
      this->pool_desc_,
      nullptr, // alpha
      this->input_desc_,
      X.template data<T, Context>(),
      nullptr, // beta
      ctx()->workspace()->template data<Context>(buffer_size, "BufferKernel"),
      this->output_desc_,
      Y->Reshape(out_shape_)->template mutable_data<T, Context>(),
      ctx()->workspace()->template data<Context>(scratch_size),
      scratch_size));
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
