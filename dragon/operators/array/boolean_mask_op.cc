#include "dragon/operators/array/boolean_mask_op.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void BooleanMaskOp<Context>::DoRunWithType() {
  auto &X = Input(0), &X_mask = Input(1), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);

  CHECK_EQ(X.count(), X_mask.count())
      << "\nSize of mask and input should be equal.";
  CHECK(X_mask.template IsType<bool>() || X_mask.template IsType<uint8_t>())
      << "\nExcepted bool or uint8 mask.";

  // Select the index of values matching the criteria.
  auto* X_index = Output("X_index")->Reshape({X.count() + 1});
  // The first "N" indices are valid.
  int N;
  kernels::Flagged(
      X.count(),
      (const uint8_t*)X_mask.template raw_data<Context>(),
      X_index->template mutable_data<int, Context>(),
      &N,
      ctx());

  // Select the values according to the flat indices.
  kernels::BooleanMask(
      N,
      X_index->Reshape({N})->template data<int, Context>(),
      X.template data<T, Context>(),
      Y->Reshape({N})->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void BooleanMaskGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  auto &X_spec = Input("X_spec"), &X_index = Input("X_index");

  math::Set(
      dX->count(),
      convert::To<T>(0.f),
      dX->ReshapeLike(X_spec)->template mutable_data<T, Context>(),
      ctx());

  kernels::BooleanMaskGrad(
      X_index.count(),
      X_index.template data<int, Context>(),
      dY.template data<T, Context>(),
      dX->template mutable_data<T, Context>(),
      ctx());
}

DEPLOY_CPU_OPERATOR(BooleanMask);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(BooleanMask);
#endif

DEPLOY_CPU_OPERATOR(BooleanMaskGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(BooleanMaskGradient);
#endif

OPERATOR_SCHEMA(BooleanMask)
    /* X, X_mask */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(BooleanMaskGradient)
    /* dY */
    .NumInputs(1)
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
        vector<string>({GO(0)}),
        vector<string>({GI(0)}));
  }
};

} // namespace

REGISTER_GRADIENT(BooleanMask, GradientMaker);

} // namespace dragon
