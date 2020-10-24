#include "dragon/operators/activation/swish_op.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void SwishOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  kernel::Swish(
      X.count(),
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void SwishOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void SwishGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &Y = Input(1);
  auto &dY = Input(2), *dX = Output(0);
  kernel::SwishGrad(
      X.count(),
      dY.template data<T, Context>(),
      X.template data<T, Context>(),
      Y.template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void SwishGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Swish);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Swish);
#endif

DEPLOY_CPU_OPERATOR(SwishGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SwishGradient);
#endif

OPERATOR_SCHEMA(Swish)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(SwishGradient)
    /* X, Y, dY */
    .NumInputs(3)
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
        vector<string>({I(0), O(0), GO(0)}),
        vector<string>({GI(0)}));
  }
};

} // namespace

REGISTER_GRADIENT(Swish, GradientMaker);

} // namespace dragon
