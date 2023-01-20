#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/vision/resize_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLResizeOp<Context>::DoRunWithType() {
  ComputeOutShape();
  auto &X = Input(0), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);
  Output("X_sizes")->template CopyFrom<int64_t>(in_dims_);
  Output("Y_sizes")->template CopyFrom<int64_t>(out_dims_);

  auto mode = CNNL_INTERP_LINEAR;
  auto coord_mode = CNNL_INTERP_COORDINATE_TRANSFORMATION_ALGO0;
  if (mode_ == "NEAREST") {
    if (out_dims_.size() == 1 || out_dims_.size() == 2) {
      mode = CNNL_INTERP_NEAREST;
      coord_mode = CNNL_INTERP_COORDINATE_TRANSFORMATION_ALGO3;
    } else {
      LOG(FATAL) << "ResizeNearest" << out_dims_.size()
                 << "d is not supported.";
    }
  } else if (mode_ == "LINEAR") {
    coord_mode = align_corners_ > 0
        ? CNNL_INTERP_COORDINATE_TRANSFORMATION_ALGO2
        : CNNL_INTERP_COORDINATE_TRANSFORMATION_ALGO0;
    if (out_dims_.size() == 1) {
      mode = CNNL_INTERP_LINEAR;
    } else if (out_dims_.size() == 2) {
      mode = CNNL_INTERP_BILINEAR;
    } else if (out_dims_.size() == 3) {
      mode = CNNL_INTERP_TRILINEAR;
    } else {
      LOG(FATAL) << "ResizeLinear" << out_dims_.size() << "d is not supported.";
    }
  }

  auto data_format_v2 = data_format();
  data_format_v2 = out_dims_.size() == 1 ? "NLC" : data_format_v2;
  CNNLSetTensorDesc<T>(input_desc_, X.dims(), data_format_v2);
  CNNLSetTensorDesc<T>(output_desc_, out_shape_, data_format_v2);
  CNNL_CHECK(cnnlSetInterpDescriptor(resize_desc_, mode, coord_mode));
  CNNL_CHECK(cnnlSetInterpDescriptorEx(
      resize_desc_,
      input_desc_,
      CNNL_INTERP_FLOOR,
      nullptr,
      nullptr,
      -0.75f,
      false));
  CNNL_CHECK(cnnlInterp_v3(
      ctx()->cnnl_handle(),
      resize_desc_,
      input_desc_,
      X.template data<T, Context>(),
      output_desc_,
      Y->Reshape(out_shape_)->template mutable_data<T, Context>()));
}

template <class Context>
template <typename T>
void CNNLResizeGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0)->ReshapeLike(Input("X_spec"));
  Input("X_sizes").template CopyTo<int64_t>(in_dims_);
  Input("Y_sizes").template CopyTo<int64_t>(out_dims_);

  auto mode = CNNL_INTERP_BACKWARD_LINEAR;
  bool align_corners = false, align_center = false;
  if (mode_ == "NEAREST") {
    if (out_dims_.size() == 1 || out_dims_.size() == 2) {
      mode = CNNL_INTERP_BACKWARD_NEAREST;
      align_corners = align_center = false;
    } else {
      LOG(FATAL) << "ResizeNearest" << out_dims_.size()
                 << "d is not supported.";
    }
  } else if (mode_ == "LINEAR") {
    align_corners = align_corners_ > 0;
    align_center = align_corners_ == 0;
    if (out_dims_.size() == 1) {
      mode = CNNL_INTERP_BACKWARD_LINEAR;
    } else if (out_dims_.size() == 2) {
      mode = CNNL_INTERP_BACKWARD_BILINEAR;
    } else if (out_dims_.size() == 3) {
      mode = CNNL_INTERP_BACKWARD_TRILINEAR;
    } else {
      LOG(FATAL) << "ResizeLinear" << out_dims_.size() << "d is not supported.";
    }
  }

  auto data_format_v2 = data_format();
  data_format_v2 = out_dims_.size() == 1 ? "NLC" : data_format_v2;
  CNNLSetTensorDesc<float>(input_desc_, dY.dims(), data_format_v2);
  CNNLSetTensorDesc<float>(output_desc_, dX->dims(), data_format_v2);

  auto* dy = dY.template data<T, Context>();
  auto* dx = dX->template mutable_data<T, Context>();
  float *dy_acc = nullptr, *dx_acc = nullptr;
  if (TypeMeta::Id<T>() != TypeMeta::Id<float>()) {
    dy_acc = ctx()->workspace()->template data<float, Context>(
        dY.count() + dX->count());
    dx_acc = dy_acc + dY.count();
  }

  if (dy_acc != nullptr) math::Cast(dY.count(), dy, dy_acc, ctx());
  CNNL_CHECK(cnnlInterpBackward_v2(
      ctx()->cnnl_handle(),
      align_corners,
      align_center,
      mode,
      nullptr,
      true,
      input_desc_,
      dy_acc != nullptr ? dy_acc : reinterpret_cast<const float*>(dy),
      output_desc_,
      dx_acc != nullptr ? dx_acc : reinterpret_cast<float*>(dx)));
  if (dx_acc != nullptr) math::Cast(dX->count(), dx_acc, dx, ctx());
}

DEPLOY_CNNL_OPERATOR(Resize);
DEPLOY_CNNL_OPERATOR(ResizeGradient);

} // namespace dragon

#endif // USE_MLU
