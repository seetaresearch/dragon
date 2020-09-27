#include "dragon/core/workspace.h"
#include "dragon/operators/array/select_ops.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void MaskedSelectOp<Context>::DoRunWithType() {
  auto &X = Input(0), &X_mask = Input(1), *Y = Output(0);

  CHECK_EQ(X.count(), X_mask.count())
      << "\nSize of mask and input should be equal.";
  CHECK(X_mask.template IsType<bool>() || X_mask.template IsType<uint8_t>())
      << "\nExcepted bool or uint8 mask.";

  // Store for the gradient calculation
  STORE_INPUT_SPEC(0);
  auto* X_index = Buffer("X_index")->Reshape({X.count() + 1});

  // Select the index of values matching the criteria
  // The first ``num_selected`` indices are valid
  int num_selected;
  kernel::Flagged(
      X.count(),
      (const uint8_t*)X_mask.template raw_data<Context>(),
      X_index->template mutable_data<int, Context>(),
      &num_selected,
      ctx());

  // Select the values according to the flat indices
  kernel::MaskedSelect(
      num_selected,
      X_index->Reshape({num_selected})->template data<int, Context>(),
      X.template data<T, Context>(),
      Y->Reshape({num_selected})->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void MaskedSelectOp<Context>::RunOnDevice() {
  DispatchHelper<FullTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void MaskedSelectGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0), *X_index = Buffer("X_index");
  dX->ReshapeLike(RESTORE_INPUT_SPEC(0));

  kernel::MaskedSelectGrad(
      dX->count(),
      X_index->count(),
      X_index->template data<int, Context>(),
      dY.template data<T, Context>(),
      dX->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void MaskedSelectGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(MaskedSelect);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(MaskedSelect);
#endif

DEPLOY_CPU_OPERATOR(MaskedSelectGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(MaskedSelectGradient);
#endif

OPERATOR_SCHEMA(MaskedSelect)
    /* X, M */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(MaskedSelectGradient)
    /* dY */
    .NumInputs(1)
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
        vector<string>({GO(0)}),
        vector<string>({GI(0)}));
  }
};

} // namespace

REGISTER_GRADIENT(MaskedSelect, GradientMaker);

} // namespace dragon
