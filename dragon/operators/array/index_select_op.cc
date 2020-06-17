#include "dragon/core/workspace.h"
#include "dragon/operators/array/select_ops.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

#define CANONICALIZE_AXES_WITH_TENSOR(tensor)                                 \
  CANONICALIZE_AXIS_WITH_TENSOR(tensor);                                      \
  auto num_axes = OpArg<int64_t>("num_axes", 1);                              \
  if (num_axes < 0) {                                                         \
    num_axes = tensor.ndim() - axis;                                          \
  } else if (num_axes == 0) {                                                 \
    num_axes = 1;                                                             \
  }                                                                           \
  CHECK(axis + num_axes <= tensor.ndim())                                     \
      << "\nInvalid number of axes. Got " << num_axes << ", excepted in [1, " \
      << tensor.ndim() - axis << "]."

template <class Context>
template <typename T>
void IndexSelectOp<Context>::DoRunWithType() {
  auto &X = Input(0), &X_index = Input(1), *Y = Output(0);
  CANONICALIZE_AXES_WITH_TENSOR(X);

  CHECK_GT(X_index.count(), 0) << "\nLength of indices must > 0.";
  CHECK(XIsType(X_index, int64_t)) << "\nType of index should be int64.";

  vec64_t X_dims(X.dims());
  vec64_t Y_dims(X_dims.begin(), X_dims.begin() + axis);
  Y_dims.insert(Y_dims.end(), X_index.dims().begin(), X_index.dims().end());
  Y_dims.insert(Y_dims.end(), X_dims.begin() + axis + num_axes, X_dims.end());

  // Store for the gradient calculation
  STORE_INPUT_SPEC(0);
  Buffer("X_index")->ReshapeLike(X_index)->CopyFrom(X_index, ctx());

  kernel::IndexSelect(
      X.count(0, axis),
      X.count(axis + num_axes),
      X.count(axis, axis + num_axes),
      X_index.count(),
      X_index.template data<int64_t, Context>(),
      X.template data<T, Context>(),
      Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void IndexSelectOp<Context>::RunOnDevice() {
  DispatchHelper<AllTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void IndexSelectGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0), *X_index = Buffer("X_index");
  dX->ReshapeLike(RESTORE_INPUT_SPEC(0));
  CANONICALIZE_AXES_WITH_TENSOR((*dX));

  // Reset the accumulating gradient
  math::Set(
      dX->count(),
      cast::to<T>(0.f),
      dX->template mutable_data<T, Context>(),
      ctx());

  kernel::IndexSelectGrad(
      dX->count(0, axis),
      dX->count(axis + num_axes),
      dX->count(axis, axis + num_axes),
      X_index->count(),
      X_index->template data<int64_t, Context>(),
      dY.template data<T, Context>(),
      dX->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void IndexSelectGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU(IndexSelect);
#ifdef USE_CUDA
DEPLOY_CUDA(IndexSelect);
#endif

DEPLOY_CPU(IndexSelectGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(IndexSelectGradient);
#endif

OPERATOR_SCHEMA(IndexSelect)
    /* X, I */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(IndexSelectGradient)
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

REGISTER_GRADIENT(IndexSelect, GradientMaker);

#undef CANONICALIZE_AXES_WITH_TENSOR

} // namespace dragon
