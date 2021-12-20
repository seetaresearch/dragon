#include "dragon/operators/vision/roi_align_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void RoiAlignOp<Context>::DoRunWithType() {
  auto &X = Input(0), &RoI = Input(1), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);
  Y->Reshape({RoI.dim(0), X.dim(1), out_h_, out_w_});

  kernels::RoiAlign(
      X.dim(1),
      X.dim(2),
      X.dim(3),
      out_h_,
      out_w_,
      RoI.dim(0),
      spatial_scale_,
      sampling_ratio_,
      X.template data<T, Context>(),
      RoI.template data<float, Context>(),
      Y->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void RoiAlignOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void RoiAlignGradientOp<Context>::DoRunWithType() {
  auto &RoI = Input(0), &dY = Input(1);
  auto* dX = Output(0)->ReshapeLike(Input("X_spec"));

  auto* dx = dX->template mutable_data<T, Context>();
  auto* dx_acc = (TypeMeta::Id<T>() == TypeMeta::Id<float>())
      ? (float*)nullptr
      : ctx()->workspace()->template data<float, Context>(dX->count());

  // Empty gradient
  math::Set(
      dX->count(),
      0.f,
      dx_acc != nullptr ? dx_acc : reinterpret_cast<float*>(dx),
      ctx());

  // Accumulate to dX
  kernels::RoiAlignGrad(
      dX->dim(1),
      dX->dim(2),
      dX->dim(3),
      out_h_,
      out_w_,
      RoI.dim(0),
      spatial_scale_,
      sampling_ratio_,
      dY.template data<T, Context>(),
      RoI.template data<float, Context>(),
      dx_acc != nullptr ? dx_acc : reinterpret_cast<float*>(dx),
      ctx());

  // Convert to dX if necessary
  if (dx_acc != nullptr) {
    math::Cast(dX->count(), dx_acc, dx, ctx());
  }
}

template <class Context>
void RoiAlignGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(1));
}

DEPLOY_CPU_OPERATOR(RoiAlign);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(RoiAlign);
#endif

DEPLOY_CPU_OPERATOR(RoiAlignGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(RoiAlignGradient);
#endif

OPERATOR_SCHEMA(RoiAlign)
    /* X, RoI */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(RoiAlignGradient)
    /* RoI, dY */
    .NumInputs(2)
    /* dX */
    .NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  void CreateGradientDefs() override {
    AddGradientDef(
        def().type() + "Gradient",
        "",
        vector<string>({I(1), GO(0)}),
        vector<string>({GI(0)}));
  }
};

} // namespace

REGISTER_GRADIENT(RoiAlign, GradientMaker);

} // namespace dragon
