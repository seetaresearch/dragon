#include "dragon/operators/vision/roi_pool_op.h"
#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void RoiPoolOp<Context>::DoRunWithType() {
  auto &X = Input(0), &B = Input(1), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);

  auto Y_dims = vec64_t({B.dim(0), out_h_, out_w_});
  auto C = data_format() == "NCHW" ? X.dim(1) : X.dim(-1);
  Y_dims.insert(data_format() == "NCHW" ? Y_dims.begin() + 1 : Y_dims.end(), C);

  kernels::RoiPool(
      X.dim(1),
      X.dim(2),
      X.dim(3),
      out_h_,
      out_w_,
      B.dim(0),
      spatial_scale_,
      X.template data<T, Context>(),
      B.template data<float, Context>(),
      Output("Y_mask")->Reshape(Y_dims)->template mutable_data<int, Context>(),
      Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void RoiPoolGradientOp<Context>::DoRunWithType() {
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
  kernels::RoiPoolGrad(
      dX->dim(1),
      dX->dim(2),
      dX->dim(3),
      out_h_,
      out_w_,
      B.dim(0),
      spatial_scale_,
      dY.template data<T, Context>(),
      B.template data<float, Context>(),
      const_cast<int*>(Input("Y_mask").template data<int, Context>()),
      dx_acc != nullptr ? dx_acc : reinterpret_cast<float*>(dx),
      ctx());

  // Convert to dX.
  if (dx_acc != nullptr) {
    math::Cast(dX->count(), dx_acc, dx, ctx());
  }
}

DEPLOY_CPU_OPERATOR(RoiPool);
DEPLOY_CPU_OPERATOR(RoiPoolGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(RoiPool);
DEPLOY_CUDA_OPERATOR(RoiPoolGradient);
#endif

OPERATOR_SCHEMA(RoiPool)
    /* X, Box */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(RoiPoolGradient)
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

REGISTER_GRADIENT(RoiPool, GradientMaker);

} // namespace dragon
