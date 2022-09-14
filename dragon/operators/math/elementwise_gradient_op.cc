#include "dragon/kernels/math/op_kernels.h"
#include "dragon/operators/math/elementwise_gradient_impl.h"
#include "dragon/operators/math/elementwise_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void AbsGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  math::Sign(
      X.count(),
      X.template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
  math::Mul(
      X.count(),
      dY.template data<T, Context>(),
      dX->template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void CosGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  kernels::CosGrad(
      X.count(),
      dY.template data<T, Context>(),
      X.template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void ExpGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  math::Mul(
      Y.count(),
      dY.template data<T, Context>(),
      Y.template data<T, Context>(),
      dX->ReshapeLike(Y)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void LogGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  math::Div(
      X.count(),
      dY.template data<T, Context>(),
      X.template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void NegGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  math::Neg(
      dY.count(),
      dY.template data<T, Context>(),
      dX->ReshapeLike(dY)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void ReciprocalGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  kernels::ReciprocalGrad(
      Y.count(),
      dY.template data<T, Context>(),
      Y.template data<T, Context>(),
      dX->ReshapeLike(Y)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void RsqrtGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  kernels::RsqrtGrad(
      Y.count(),
      dY.template data<T, Context>(),
      Y.template data<T, Context>(),
      dX->ReshapeLike(Y)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void SignGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  math::Set(
      dY.count(),
      convert::To<T>(0.f),
      dX->ReshapeLike(dY)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void SinGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  kernels::SinGrad(
      X.count(),
      dY.template data<T, Context>(),
      X.template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void SqrtGradientOp<Context>::DoRunWithType() {
  auto &Y = Input(0), &dY = Input(1), *dX = Output(0);
  math::Div(
      Y.count(),
      dY.template data<T, Context>(),
      Y.template data<T, Context>(),
      dX->ReshapeLike(Y)->template mutable_data<T, Context>(),
      ctx());
  math::Scale(
      Y.count(),
      0.5f,
      dX->template data<T, Context>(),
      dX->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void SquareGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  math::Mul(
      X.count(),
      dY.template data<T, Context>(),
      X.template data<T, Context>(),
      dX->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
  math::Scale(
      X.count(),
      2.f,
      dX->template data<T, Context>(),
      dX->template mutable_data<T, Context>(),
      ctx());
}

#define DISPATCH_WITH_TENSOR_TYPES(name, tensor_types, X_ref) \
  template <class Context>                                    \
  void name##Op<Context>::RunOnDevice() {                     \
    DispatchHelper<tensor_types>::Call(this, X_ref);          \
  }

DISPATCH_WITH_TENSOR_TYPES(AbsGradient, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(CosGradient, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(ExpGradient, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(LogGradient, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(NegGradient, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(ReciprocalGradient, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(RsqrtGradient, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(SignGradient, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(SinGradient, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(SqrtGradient, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(SquareGradient, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(AddGradient, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(SubGradient, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(MulGradient, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(DivGradient, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(MaximumGradient, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(MinimumGradient, dtypes::Floating, Input(0));
#undef DISPATCH_WITH_TENSOR_TYPES

DEPLOY_CPU_OPERATOR(AbsGradient);
DEPLOY_CPU_OPERATOR(CosGradient);
DEPLOY_CPU_OPERATOR(ExpGradient);
DEPLOY_CPU_OPERATOR(LogGradient);
DEPLOY_CPU_OPERATOR(NegGradient);
DEPLOY_CPU_OPERATOR(ReciprocalGradient);
DEPLOY_CPU_OPERATOR(RsqrtGradient);
DEPLOY_CPU_OPERATOR(SignGradient);
DEPLOY_CPU_OPERATOR(SinGradient);
DEPLOY_CPU_OPERATOR(SqrtGradient);
DEPLOY_CPU_OPERATOR(SquareGradient);
DEPLOY_CPU_OPERATOR(AddGradient);
DEPLOY_CPU_OPERATOR(SubGradient);
DEPLOY_CPU_OPERATOR(MulGradient);
DEPLOY_CPU_OPERATOR(DivGradient);
DEPLOY_CPU_OPERATOR(MaximumGradient);
DEPLOY_CPU_OPERATOR(MinimumGradient);

#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(AbsGradient);
DEPLOY_CUDA_OPERATOR(CosGradient);
DEPLOY_CUDA_OPERATOR(ExpGradient);
DEPLOY_CUDA_OPERATOR(LogGradient);
DEPLOY_CUDA_OPERATOR(NegGradient);
DEPLOY_CUDA_OPERATOR(ReciprocalGradient);
DEPLOY_CUDA_OPERATOR(RsqrtGradient);
DEPLOY_CUDA_OPERATOR(SignGradient);
DEPLOY_CUDA_OPERATOR(SinGradient);
DEPLOY_CUDA_OPERATOR(SqrtGradient);
DEPLOY_CUDA_OPERATOR(SquareGradient);
DEPLOY_CUDA_OPERATOR(AddGradient);
DEPLOY_CUDA_OPERATOR(SubGradient);
DEPLOY_CUDA_OPERATOR(MulGradient);
DEPLOY_CUDA_OPERATOR(DivGradient);
DEPLOY_CUDA_OPERATOR(MaximumGradient);
DEPLOY_CUDA_OPERATOR(MinimumGradient);
#endif

#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(AbsGradient, AbsGradient);
DEPLOY_MPS_OPERATOR(CosGradient, CosGradient);
DEPLOY_MPS_OPERATOR(ExpGradient, ExpGradient);
DEPLOY_MPS_OPERATOR(LogGradient, LogGradient);
DEPLOY_MPS_OPERATOR(NegGradient, NegGradient);
DEPLOY_MPS_OPERATOR(ReciprocalGradient, ReciprocalGradient);
DEPLOY_MPS_OPERATOR(RsqrtGradient, RsqrtGradient);
DEPLOY_MPS_OPERATOR(SignGradient, SignGradient);
DEPLOY_MPS_OPERATOR(SinGradient, SinGradient);
DEPLOY_MPS_OPERATOR(SqrtGradient, SqrtGradient);
DEPLOY_MPS_OPERATOR(SquareGradient, SquareGradient);
DEPLOY_MPS_OPERATOR(AddGradient, AddGradient);
DEPLOY_MPS_OPERATOR(SubGradient, SubGradient);
DEPLOY_MPS_OPERATOR(MulGradient, MulGradient);
DEPLOY_MPS_OPERATOR(DivGradient, DivGradient);
#endif

OPERATOR_SCHEMA(AbsGradient).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(CosGradient).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(ExpGradient).NumInputs(2).NumOutputs(1).AllowInplace({{1, 0}});
OPERATOR_SCHEMA(LogGradient).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(NegGradient).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(ReciprocalGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}});
OPERATOR_SCHEMA(RsqrtGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}});
OPERATOR_SCHEMA(SignGradient).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(SinGradient).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(SqrtGradient).NumInputs(2).NumOutputs(1).AllowInplace({{1, 0}});
OPERATOR_SCHEMA(SquareGradient).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(AddGradient)
    .NumInputs(1)
    .NumOutputs(2)
    .AllowInplace({{0, 0}, {0, 1}});
OPERATOR_SCHEMA(SubGradient)
    .NumInputs(1)
    .NumOutputs(2)
    .AllowInplace({{0, 0}, {0, 1}});
OPERATOR_SCHEMA(MulGradient)
    .NumInputs(3)
    .NumOutputs(2)
    .AllowInplace({{2, 0}, {2, 1}});
OPERATOR_SCHEMA(DivGradient)
    .NumInputs(3)
    .NumOutputs(2)
    .AllowInplace({{2, 0}, {2, 1}});
OPERATOR_SCHEMA(MaximumGradient).NumInputs(3).NumOutputs(2);
OPERATOR_SCHEMA(MinimumGradient).NumInputs(3).NumOutputs(2);

REGISTER_GRADIENT(Abs, GenericGradientMaker);
REGISTER_GRADIENT(Cos, GenericGradientMaker);
REGISTER_GRADIENT(Exp, InplaceGradientMaker);
REGISTER_GRADIENT(Log, GenericGradientMaker);
REGISTER_GRADIENT(Neg, SimpleGradientMaker);
REGISTER_GRADIENT(Reciprocal, InplaceGradientMaker);
REGISTER_GRADIENT(Rsqrt, InplaceGradientMaker);
REGISTER_GRADIENT(Sign, SimpleGradientMaker);
REGISTER_GRADIENT(Sin, GenericGradientMaker);
REGISTER_GRADIENT(Sqrt, InplaceGradientMaker);
REGISTER_GRADIENT(Square, GenericGradientMaker);
REGISTER_GRADIENT(Add, SimpleGradientMaker);
REGISTER_GRADIENT(Sub, SimpleGradientMaker);
REGISTER_GRADIENT(Mul, GenericGradientMaker);
REGISTER_GRADIENT(Div, GenericGradientMaker);
REGISTER_GRADIENT(Maximum, GenericGradientMaker);
REGISTER_GRADIENT(Minimum, GenericGradientMaker);

} // namespace dragon
