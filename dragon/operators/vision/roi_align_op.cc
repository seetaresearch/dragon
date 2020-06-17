#include "dragon/operators/vision/roi_align_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void RoiAlignOp<Context>::DoRunWithType() {
  auto &X = Input(0), &RoI = Input(1), *Y = Output(0);
  Y->Reshape({RoI.dim(0), X.dim(1), out_h_, out_w_});

  // Store for the gradient calculation
  STORE_INPUT_SPEC(0);

  kernel::RoiAlign(
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
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void RoiAlignGradientOp<Context>::DoRunWithType() {
  auto &RoI = Input(0), &dY = Input(1);
  auto* dX = Output(0)->ReshapeLike(RESTORE_INPUT_SPEC(0));

  math::Set(
      dX->count(),
      cast::to<T>(0.f),
      dX->template mutable_data<T, Context>(),
      ctx());

  kernel::RoiAlignGrad(
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
      dX->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void RoiAlignGradientOp<Context>::DoRunWithTypeAndCast() {
  auto &RoI = Input(0), &dY = Input(1);
  auto* dX = Output(0)->ReshapeLike(RESTORE_INPUT_SPEC(0));

  auto* scratch = ws()->template data<float, Context>({dX->count()})[0];
  math::Set(dX->count(), 0.f, scratch, ctx());
  kernel::RoiAlignGrad(
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
      scratch,
      ctx());
  kernel::Cast(
      dX->count(), scratch, dX->template mutable_data<T, Context>(), ctx());
}

template <class Context>
void RoiAlignGradientOp<Context>::RunOnDevice() {
  if (XIsType(Input(1), float16)) {
    DoRunWithTypeAndCast<float16>();
  } else if (XIsType(Input(1), float)) {
    DoRunWithType<float>();
  } else if (XIsType(Input(1), double)) {
    DoRunWithTypeAndCast<double>();
  } else {
    LOG(FATAL) << TypeString(Input(1), {"float16", "float32", "float64"});
  };
}

DEPLOY_CPU(RoiAlign);
#ifdef USE_CUDA
DEPLOY_CUDA(RoiAlign);
#endif

DEPLOY_CPU(RoiAlignGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(RoiAlignGradient);
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
  vector<OperatorDef> MakeDef() override {
    return SingleDef(
        def.type() + "Gradient",
        "",
        vector<string>({I(1), GO(0)}),
        vector<string>({GI(0)}));
  }
};

} // namespace

REGISTER_GRADIENT(RoiAlign, GradientMaker);

} // namespace dragon
