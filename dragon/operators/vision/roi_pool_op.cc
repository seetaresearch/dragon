#include "dragon/operators/vision/roi_pool_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void RoiPoolOp<Context>::DoRunWithType() {
  auto &X = Input(0), &RoI = Input(1), *Y = Output(0);
  Y->Reshape({RoI.dim(0), X.dim(1), out_h_, out_w_});
  Buffer("Y_mask")->ReshapeLike(*Y);

  // Store for the gradient calculation
  STORE_INPUT_SPEC(0);

  kernel::RoiPool(
      X.dim(1),
      X.dim(2),
      X.dim(3),
      out_h_,
      out_w_,
      RoI.dim(0),
      spatial_scale_,
      X.template data<T, Context>(),
      RoI.template data<float, Context>(),
      Buffer("Y_mask")->template mutable_data<int, Context>(),
      Y->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void RoiPoolOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void RoiPoolGradientOp<Context>::DoRunWithType() {
  auto &RoI = Input(0), &dY = Input(1);
  auto* dX = Output(0)->ReshapeLike(RESTORE_INPUT_SPEC(0));

  math::Set(
      dX->count(),
      cast::to<T>(0.f),
      dX->template mutable_data<T, Context>(),
      ctx());

  kernel::RoiPoolGrad(
      dX->dim(1),
      dX->dim(2),
      dX->dim(3),
      out_h_,
      out_w_,
      RoI.dim(0),
      spatial_scale_,
      dY.template data<T, Context>(),
      RoI.template data<float, Context>(),
      Buffer("Y_mask")->template data<int, Context>(),
      dX->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void RoiPoolGradientOp<Context>::DoRunWithTypeAndCast() {
  auto &RoI = Input(0), &dY = Input(1);
  auto* dX = Output(0)->ReshapeLike(RESTORE_INPUT_SPEC(0));

  auto* scratch = ws()->template data<float, Context>({dX->count()})[0];
  math::Set(dX->count(), 0.f, scratch, ctx());

  kernel::RoiPoolGrad(
      dX->dim(1),
      dX->dim(2),
      dX->dim(3),
      out_h_,
      out_w_,
      RoI.dim(0),
      spatial_scale_,
      dY.template data<T, Context>(),
      RoI.template data<float, Context>(),
      Buffer("Y_mask")->template data<int, Context>(),
      scratch,
      ctx());

  kernel::Cast(
      dX->count(), scratch, dX->template mutable_data<T, Context>(), ctx());
}

template <class Context>
void RoiPoolGradientOp<Context>::RunOnDevice() {
  if (XIsType(Input(1), float16)) {
    DoRunWithTypeAndCast<float16>();
  } else if (XIsType(Input(1), float)) {
    DoRunWithType<float>();
  } else if (XIsType(Input(1), double)) {
    DoRunWithTypeAndCast<double>();
  } else {
    LOG(FATAL) << MessageForUnsupported(
        types::to_string(Input(1).meta()), {"float16", "float32", "float64"});
  };
}

DEPLOY_CPU(RoiPool);
#ifdef USE_CUDA
DEPLOY_CUDA(RoiPool);
#endif

DEPLOY_CPU(RoiPoolGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(RoiPoolGradient);
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
  vector<OperatorDef> MakeDef() override {
    return SingleDef(
        def.type() + "Gradient",
        "",
        vector<string>({I(1), GO(0)}),
        vector<string>({GI(0)}));
  }
};

} // namespace

REGISTER_GRADIENT(RoiPool, GradientMaker);

} // namespace dragon
