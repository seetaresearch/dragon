#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/vision/roi_align_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLRoiAlignOp<Context>::DoRunWithType() {
  auto &X = Input(0), &B = Input(1), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);

  auto Y_dims = vec64_t({B.dim(0), out_h_, out_w_});
  auto C = data_format() == "NCHW" ? X.dim(1) : X.dim(-1);
  Y_dims.insert(data_format() == "NCHW" ? Y_dims.begin() + 1 : Y_dims.end(), C);

  CNNLSetTensorDesc<T>(input_desc_, X.dims(), data_format());
  CNNLSetTensorDesc<T>(output_desc_, Y_dims, data_format());
  CNNLSetTensorDesc<T>(boxes_desc_, B.dims());

  auto* boxes = B.template data<float, Context>();
  T* boxes_cast = nullptr;
  if (TypeMeta::Id<T>() != TypeMeta::Id<float>()) {
    boxes_cast = ctx()->workspace()->template data<T, Context>(B.count());
    math::Cast(B.count(), boxes, boxes_cast, ctx());
  }

  CNNL_CHECK(cnnlSetRoiAlignDescriptor_v2(
      pool_desc_,
      out_h_,
      out_w_,
      sampling_ratio_,
      spatial_scale_,
      1,
      aligned_ > 0));
  CNNL_CHECK(cnnlRoiAlign_v2(
      ctx()->cnnl_handle(),
      pool_desc_,
      input_desc_,
      X.template data<T, Context>(),
      boxes_desc_,
      boxes_cast != nullptr ? boxes_cast : reinterpret_cast<const T*>(boxes),
      output_desc_,
      Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
      nullptr,
      nullptr,
      nullptr,
      nullptr));
}

template <class Context>
template <typename T>
void CNNLRoiAlignGradientOp<Context>::DoRunWithType() {
  auto &B = Input(0), &dY = Input(1);
  auto* dX = Output(0)->ReshapeLike(Input("X_spec"));

  CNNLSetTensorDesc<float>(this->input_desc_, dX->dims(), data_format());
  CNNLSetTensorDesc<float>(this->output_desc_, dY.dims(), data_format());
  CNNLSetTensorDesc<float>(this->boxes_desc_, B.dims());

  auto* dy = dY.template data<T, Context>();
  auto* dx = dX->template mutable_data<T, Context>();
  float *dy_acc = nullptr, *dx_acc = nullptr;
  if (TypeMeta::Id<T>() != TypeMeta::Id<float>()) {
    const auto scratch_count = dY.count() + dX->count();
    dy_acc = ctx()->workspace()->template data<float, Context>(scratch_count);
    dx_acc = dy_acc + dY.count();
  }

  if (dy_acc != nullptr) math::Cast(dY.count(), dy, dy_acc, ctx());
  CNNL_CHECK(cnnlRoiAlignBackward_v2(
      ctx()->cnnl_handle(),
      this->output_desc_,
      dy_acc != nullptr ? dy_acc : reinterpret_cast<const float*>(dy),
      this->boxes_desc_,
      B.template data<float, Context>(),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      this->spatial_scale_,
      this->sampling_ratio_,
      this->aligned_ > 0,
      1,
      this->input_desc_,
      dx_acc != nullptr ? dx_acc : reinterpret_cast<float*>(dx)));
  if (dx_acc != nullptr) math::Cast(dX->count(), dx_acc, dx, ctx());
}

DEPLOY_CNNL_OPERATOR(RoiAlign);
DEPLOY_CNNL_OPERATOR(RoiAlignGradient);

} // namespace dragon

#endif // USE_MLU
