#include "dragon/operators/vision/roi_pool_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void RoiPoolOp<Context>::DoRunWithType() {
  auto &X = Input(0), &RoI = Input(1), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);
  Y->Reshape({RoI.dim(0), X.dim(1), out_h_, out_w_});
  auto* Y_mask = Output("Y_mask")->ReshapeLike(*Y);

  kernels::RoiPool(
      X.dim(1),
      X.dim(2),
      X.dim(3),
      out_h_,
      out_w_,
      RoI.dim(0),
      spatial_scale_,
      X.template data<T, Context>(),
      RoI.template data<float, Context>(),
      Y_mask->template mutable_data<int, Context>(),
      Y->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void RoiPoolOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void RoiPoolGradientOp<Context>::DoRunWithType() {
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
  kernels::RoiPoolGrad(
      dX->dim(1),
      dX->dim(2),
      dX->dim(3),
      out_h_,
      out_w_,
      RoI.dim(0),
      spatial_scale_,
      dY.template data<T, Context>(),
      RoI.template data<float, Context>(),
      const_cast<int*>(Input("Y_mask").template data<int, Context>()),
      dx_acc != nullptr ? dx_acc : reinterpret_cast<float*>(dx),
      ctx());

  // Convert to dX if necessary
  if (dx_acc != nullptr) {
    math::Cast(dX->count(), dx_acc, dx, ctx());
  }
}

template <class Context>
void RoiPoolGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(1));
}

DEPLOY_CPU_OPERATOR(RoiPool);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(RoiPool);
#endif

DEPLOY_CPU_OPERATOR(RoiPoolGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(RoiPoolGradient);
#endif

OPERATOR_SCHEMA(RoiPool)
    /* X, RoI */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(RoiPoolGradient)
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

REGISTER_GRADIENT(RoiPool, GradientMaker);

} // namespace dragon
