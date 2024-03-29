#include "dragon/operators/vision/roi_align_op.h"
#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void RoiAlignOp<Context>::DoRunWithType() {
  auto &X = Input(0), &B = Input(1), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);

  auto Y_dims = vec64_t({B.dim(0), out_h_, out_w_});
  auto C = data_format() == "NCHW" ? X.dim(1) : X.dim(-1);
  Y_dims.insert(data_format() == "NCHW" ? Y_dims.begin() + 1 : Y_dims.end(), C);

  kernels::RoiAlign(
      X.dim(1),
      X.dim(2),
      X.dim(3),
      out_h_,
      out_w_,
      B.dim(0),
      spatial_scale_,
      sampling_ratio_,
      aligned_ > 0,
      X.template data<T, Context>(),
      B.template data<float, Context>(),
      Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void RoiAlignGradientOp<Context>::DoRunWithType() {
  auto &B = Input(0), &dY = Input(1);
  auto* dX = Output(0)->ReshapeLike(Input("X_spec"));

  auto* dx = dX->template mutable_data<T, Context>();
  auto* dx_acc = (TypeMeta::Id<T>() == TypeMeta::Id<float>())
      ? (float*)nullptr
      : ctx()->workspace()->template data<float, Context>(dX->count());

  // Empty gradient.
  math::Set(
      dX->count(),
      0.f,
      dx_acc != nullptr ? dx_acc : reinterpret_cast<float*>(dx),
      ctx());

  // Accumulate to dX.
  kernels::RoiAlignGrad(
      dX->dim(1),
      dX->dim(2),
      dX->dim(3),
      out_h_,
      out_w_,
      B.dim(0),
      spatial_scale_,
      sampling_ratio_,
      aligned_ > 0,
      dY.template data<T, Context>(),
      B.template data<float, Context>(),
      dx_acc != nullptr ? dx_acc : reinterpret_cast<float*>(dx),
      ctx());

  // Convert to dX.
  if (dx_acc != nullptr) {
    math::Cast(dX->count(), dx_acc, dx, ctx());
  }
}

DEPLOY_CPU_OPERATOR(RoiAlign);
DEPLOY_CPU_OPERATOR(RoiAlignGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(RoiAlign);
DEPLOY_CUDA_OPERATOR(RoiAlignGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(RoiAlign, RoiAlign);
#endif

OPERATOR_SCHEMA(RoiAlign)
    /* X, Box */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(RoiAlignGradient)
    /* Box, dY */
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
