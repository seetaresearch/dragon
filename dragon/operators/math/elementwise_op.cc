#include "dragon/operators/math/elementwise_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

#define DISPATCH_WITH_TENSOR_TYPES(name, tensor_types, X_ref) \
  template <class Context>                                    \
  void name##Op<Context>::RunOnDevice() {                     \
    DispatchHelper<tensor_types>::Call(this, X_ref);          \
  }

DISPATCH_WITH_TENSOR_TYPES(Neg, dtypes::Signed, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Abs, dtypes::Numerical, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Square, dtypes::Numerical, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Sign, dtypes::Numerical, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Ceil, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Floor, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Round, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Exp, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Log, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Reciprocal, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Sqrt, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Rsqrt, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Sin, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Cos, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(IsInf, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(IsNaN, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(IsFinite, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Pow, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Atan2, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Add, dtypes::Numerical, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Sub, dtypes::Numerical, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Mul, dtypes::Numerical, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Div, dtypes::Numerical, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Minimum, dtypes::Numerical, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Maximum, dtypes::Numerical, Input(0));
DISPATCH_WITH_TENSOR_TYPES(BitwiseNot, dtypes::Bitwise, Input(0));
DISPATCH_WITH_TENSOR_TYPES(BitwiseAnd, dtypes::Bitwise, Input(0));
DISPATCH_WITH_TENSOR_TYPES(BitwiseOr, dtypes::Bitwise, Input(0));
DISPATCH_WITH_TENSOR_TYPES(BitwiseXor, dtypes::Bitwise, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Not, dtypes::Generic, Input(0));
DISPATCH_WITH_TENSOR_TYPES(And, dtypes::Generic, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Or, dtypes::Generic, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Xor, dtypes::Generic, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Equal, dtypes::Generic, Input(0));
DISPATCH_WITH_TENSOR_TYPES(NotEqual, dtypes::Generic, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Less, dtypes::Generic, Input(0));
DISPATCH_WITH_TENSOR_TYPES(LessEqual, dtypes::Generic, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Greater, dtypes::Generic, Input(0));
DISPATCH_WITH_TENSOR_TYPES(GreaterEqual, dtypes::Generic, Input(0));
#undef DISPATCH_WITH_TENSOR_TYPES

#define DEFINE_SIMPLE_UNARY_OP_IMPL(name, OutputT)                    \
  template <class Context>                                            \
  template <typename T>                                               \
  void name##Op<Context>::DoRunWithType() {                           \
    auto &X = Input(0), *Y = Output(0);                               \
    math::name(                                                       \
        X.count(),                                                    \
        X.template data<T, Context>(),                                \
        Y->ReshapeLike(X)->template mutable_data<OutputT, Context>(), \
        ctx());                                                       \
  }

DEFINE_SIMPLE_UNARY_OP_IMPL(Abs, T);
DEFINE_SIMPLE_UNARY_OP_IMPL(Square, T);
DEFINE_SIMPLE_UNARY_OP_IMPL(Log, T);
DEFINE_SIMPLE_UNARY_OP_IMPL(Sin, T);
DEFINE_SIMPLE_UNARY_OP_IMPL(Cos, T);
DEFINE_SIMPLE_UNARY_OP_IMPL(IsInf, bool);
DEFINE_SIMPLE_UNARY_OP_IMPL(IsNaN, bool);
DEFINE_SIMPLE_UNARY_OP_IMPL(IsFinite, bool);
DEFINE_SIMPLE_UNARY_OP_IMPL(Not, bool);
#undef DEFINE_SIMPLE_UNARY_OP_IMPL

#define DEFINE_INPLACE_UNARY_OP_IMPL(name, OutputT)                   \
  template <class Context>                                            \
  template <typename T>                                               \
  void name##Op<Context>::DoRunWithType() {                           \
    auto &X = Input(0), *Y = Output(0, {0});                          \
    math::name(                                                       \
        X.count(),                                                    \
        X.template data<T, Context>(),                                \
        Y->ReshapeLike(X)->template mutable_data<OutputT, Context>(), \
        ctx());                                                       \
  }

template <class Context>
template <typename T>
void ReciprocalOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  math::Inv(
      X.count(),
      X.template data<T, Context>(),
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      ctx());
}

DEFINE_INPLACE_UNARY_OP_IMPL(Neg, T);
DEFINE_INPLACE_UNARY_OP_IMPL(Sign, T);
DEFINE_INPLACE_UNARY_OP_IMPL(Ceil, T);
DEFINE_INPLACE_UNARY_OP_IMPL(Floor, T);
DEFINE_INPLACE_UNARY_OP_IMPL(Round, T);
DEFINE_INPLACE_UNARY_OP_IMPL(Exp, T);
DEFINE_INPLACE_UNARY_OP_IMPL(Sqrt, T);
DEFINE_INPLACE_UNARY_OP_IMPL(Rsqrt, T);
DEFINE_INPLACE_UNARY_OP_IMPL(BitwiseNot, T);
#undef DEFINE_INPLACE_UNARY_OP_IMPL

#define DEFINE_SIMPLE_BINARY_OP_IMPL(name, OutputT)                          \
  template <class Context>                                                   \
  template <typename T>                                                      \
  void name##Op<Context>::DoRunWithType() {                                  \
    auto &A = Input(0), &B = Input(1), *Y = Output(0);                       \
    vec64_t Y_dims(A.dims());                                                \
    if (A.dims() == B.dims()) {                                              \
      math::name(                                                            \
          A.count(),                                                         \
          A.template data<T, Context>(),                                     \
          B.template data<T, Context>(),                                     \
          Y->Reshape(Y_dims)->template mutable_data<OutputT, Context>(),     \
          ctx());                                                            \
    } else if (math::utils::IsBinaryBroadcast(A.dims(), B.dims(), Y_dims)) { \
      math::name(                                                            \
          A.ndim(),                                                          \
          A.dims().data(),                                                   \
          B.ndim(),                                                          \
          B.dims().data(),                                                   \
          A.template data<T, Context>(),                                     \
          B.template data<T, Context>(),                                     \
          Y->Reshape(Y_dims)->template mutable_data<OutputT, Context>(),     \
          ctx());                                                            \
    } else {                                                                 \
      LOG(FATAL) << "Could not broadcast with: " << A.DimString() << " "     \
                 << B.DimString();                                           \
    }                                                                        \
  }

DEFINE_SIMPLE_BINARY_OP_IMPL(Pow, T);
DEFINE_SIMPLE_BINARY_OP_IMPL(Atan2, T);
DEFINE_SIMPLE_BINARY_OP_IMPL(Minimum, T);
DEFINE_SIMPLE_BINARY_OP_IMPL(Maximum, T);
DEFINE_SIMPLE_BINARY_OP_IMPL(BitwiseAnd, T);
DEFINE_SIMPLE_BINARY_OP_IMPL(BitwiseOr, T);
DEFINE_SIMPLE_BINARY_OP_IMPL(BitwiseXor, T);
DEFINE_SIMPLE_BINARY_OP_IMPL(And, bool);
DEFINE_SIMPLE_BINARY_OP_IMPL(Or, bool);
DEFINE_SIMPLE_BINARY_OP_IMPL(Xor, bool);
DEFINE_SIMPLE_BINARY_OP_IMPL(Equal, bool);
DEFINE_SIMPLE_BINARY_OP_IMPL(NotEqual, bool);
DEFINE_SIMPLE_BINARY_OP_IMPL(Less, bool);
DEFINE_SIMPLE_BINARY_OP_IMPL(LessEqual, bool);
DEFINE_SIMPLE_BINARY_OP_IMPL(Greater, bool);
DEFINE_SIMPLE_BINARY_OP_IMPL(GreaterEqual, bool);
#undef DEFINE_SIMPLE_BINARY_OP_IMPL

#define DEFINE_INPLACE_BINARY_OP_IMPL(name, OutputT)                         \
  template <class Context>                                                   \
  template <typename T>                                                      \
  void name##Op<Context>::DoRunWithType() {                                  \
    auto &A = Input(0), &B = Input(1), *Y = Output(0, {0, 1});               \
    Output("A_spec")->ReshapeLike(A);                                        \
    Output("B_spec")->ReshapeLike(B);                                        \
    vec64_t Y_dims(A.dims());                                                \
    if (A.dims() == B.dims()) {                                              \
      math::name(                                                            \
          A.count(),                                                         \
          A.template data<T, Context>(),                                     \
          B.template data<T, Context>(),                                     \
          Y->Reshape(Y_dims)->template mutable_data<OutputT, Context>(),     \
          ctx());                                                            \
    } else if (math::utils::IsBinaryBroadcast(A.dims(), B.dims(), Y_dims)) { \
      Y = Output(0, CheckOutputAliases(A, B, Output(0), Y_dims));            \
      math::name(                                                            \
          A.ndim(),                                                          \
          A.dims().data(),                                                   \
          B.ndim(),                                                          \
          B.dims().data(),                                                   \
          A.template data<T, Context>(),                                     \
          B.template data<T, Context>(),                                     \
          Y->Reshape(Y_dims)->template mutable_data<OutputT, Context>(),     \
          ctx());                                                            \
    } else {                                                                 \
      LOG(FATAL) << "Could not broadcast with shapes: " << A.DimString()     \
                 << " " << B.DimString();                                    \
    }                                                                        \
  }

DEFINE_INPLACE_BINARY_OP_IMPL(Add, T);
DEFINE_INPLACE_BINARY_OP_IMPL(Sub, T);
DEFINE_INPLACE_BINARY_OP_IMPL(Mul, T);
DEFINE_INPLACE_BINARY_OP_IMPL(Div, T);
#undef DEFINE_INPLACE_BINARY_OP_IMPL

DEPLOY_CPU_OPERATOR(Neg);
DEPLOY_CPU_OPERATOR(Abs);
DEPLOY_CPU_OPERATOR(Square);
DEPLOY_CPU_OPERATOR(Sign);
DEPLOY_CPU_OPERATOR(Ceil);
DEPLOY_CPU_OPERATOR(Floor);
DEPLOY_CPU_OPERATOR(Round);
DEPLOY_CPU_OPERATOR(Exp);
DEPLOY_CPU_OPERATOR(Log);
DEPLOY_CPU_OPERATOR(Reciprocal);
DEPLOY_CPU_OPERATOR(Sqrt);
DEPLOY_CPU_OPERATOR(Rsqrt);
DEPLOY_CPU_OPERATOR(Sin);
DEPLOY_CPU_OPERATOR(Cos);
DEPLOY_CPU_OPERATOR(IsInf);
DEPLOY_CPU_OPERATOR(IsNaN);
DEPLOY_CPU_OPERATOR(IsFinite);
DEPLOY_CPU_OPERATOR(Add);
DEPLOY_CPU_OPERATOR(Sub);
DEPLOY_CPU_OPERATOR(Mul);
DEPLOY_CPU_OPERATOR(Div);
DEPLOY_CPU_OPERATOR(Pow);
DEPLOY_CPU_OPERATOR(Atan2);
DEPLOY_CPU_OPERATOR(Minimum);
DEPLOY_CPU_OPERATOR(Maximum);
DEPLOY_CPU_OPERATOR(BitwiseNot);
DEPLOY_CPU_OPERATOR(BitwiseAnd);
DEPLOY_CPU_OPERATOR(BitwiseOr);
DEPLOY_CPU_OPERATOR(BitwiseXor);
DEPLOY_CPU_OPERATOR(Not);
DEPLOY_CPU_OPERATOR(And);
DEPLOY_CPU_OPERATOR(Or);
DEPLOY_CPU_OPERATOR(Xor);
DEPLOY_CPU_OPERATOR(Equal);
DEPLOY_CPU_OPERATOR(NotEqual);
DEPLOY_CPU_OPERATOR(Less);
DEPLOY_CPU_OPERATOR(LessEqual);
DEPLOY_CPU_OPERATOR(Greater);
DEPLOY_CPU_OPERATOR(GreaterEqual);

#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Neg);
DEPLOY_CUDA_OPERATOR(Abs);
DEPLOY_CUDA_OPERATOR(Square);
DEPLOY_CUDA_OPERATOR(Sign);
DEPLOY_CUDA_OPERATOR(Ceil);
DEPLOY_CUDA_OPERATOR(Floor);
DEPLOY_CUDA_OPERATOR(Round);
DEPLOY_CUDA_OPERATOR(Exp);
DEPLOY_CUDA_OPERATOR(Log);
DEPLOY_CUDA_OPERATOR(Reciprocal);
DEPLOY_CUDA_OPERATOR(Sqrt);
DEPLOY_CUDA_OPERATOR(Rsqrt);
DEPLOY_CUDA_OPERATOR(Sin);
DEPLOY_CUDA_OPERATOR(Cos);
DEPLOY_CUDA_OPERATOR(IsInf);
DEPLOY_CUDA_OPERATOR(IsNaN);
DEPLOY_CUDA_OPERATOR(IsFinite);
DEPLOY_CUDA_OPERATOR(Add);
DEPLOY_CUDA_OPERATOR(Sub);
DEPLOY_CUDA_OPERATOR(Mul);
DEPLOY_CUDA_OPERATOR(Div);
DEPLOY_CUDA_OPERATOR(Pow);
DEPLOY_CUDA_OPERATOR(Atan2);
DEPLOY_CUDA_OPERATOR(Minimum);
DEPLOY_CUDA_OPERATOR(Maximum);
DEPLOY_CUDA_OPERATOR(BitwiseNot);
DEPLOY_CUDA_OPERATOR(BitwiseAnd);
DEPLOY_CUDA_OPERATOR(BitwiseOr);
DEPLOY_CUDA_OPERATOR(BitwiseXor);
DEPLOY_CUDA_OPERATOR(Not);
DEPLOY_CUDA_OPERATOR(And);
DEPLOY_CUDA_OPERATOR(Or);
DEPLOY_CUDA_OPERATOR(Xor);
DEPLOY_CUDA_OPERATOR(Equal);
DEPLOY_CUDA_OPERATOR(NotEqual);
DEPLOY_CUDA_OPERATOR(Less);
DEPLOY_CUDA_OPERATOR(LessEqual);
DEPLOY_CUDA_OPERATOR(Greater);
DEPLOY_CUDA_OPERATOR(GreaterEqual);
#endif

#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(Neg, Neg);
DEPLOY_MPS_OPERATOR(Abs, Abs);
DEPLOY_MPS_OPERATOR(Square, Square);
DEPLOY_MPS_OPERATOR(Sign, Sign);
DEPLOY_MPS_OPERATOR(Ceil, Ceil);
DEPLOY_MPS_OPERATOR(Floor, Floor);
DEPLOY_MPS_OPERATOR(Round, Round);
DEPLOY_MPS_OPERATOR(Exp, Exp);
DEPLOY_MPS_OPERATOR(Log, Log);
DEPLOY_MPS_OPERATOR(Reciprocal, Reciprocal);
DEPLOY_MPS_OPERATOR(Sqrt, Sqrt);
DEPLOY_MPS_OPERATOR(Rsqrt, Rsqrt);
DEPLOY_MPS_OPERATOR(Sin, Sin);
DEPLOY_MPS_OPERATOR(Cos, Cos);
DEPLOY_MPS_OPERATOR(IsInf, IsInf);
DEPLOY_MPS_OPERATOR(IsNaN, IsNaN);
DEPLOY_MPS_OPERATOR(IsFinite, IsFinite);
DEPLOY_MPS_OPERATOR(Add, Add);
DEPLOY_MPS_OPERATOR(Sub, Sub);
DEPLOY_MPS_OPERATOR(Mul, Mul);
DEPLOY_MPS_OPERATOR(Div, Div);
DEPLOY_MPS_OPERATOR(Pow, Pow);
DEPLOY_MPS_OPERATOR(Atan2, Atan2);
DEPLOY_MPS_OPERATOR(Minimum, Minimum);
DEPLOY_MPS_OPERATOR(Maximum, Maximum);
DEPLOY_MPS_OPERATOR(BitwiseNot, BitwiseNot);
DEPLOY_MPS_OPERATOR(BitwiseAnd, BitwiseAnd);
DEPLOY_MPS_OPERATOR(BitwiseOr, BitwiseOr);
DEPLOY_MPS_OPERATOR(BitwiseXor, BitwiseXor);
DEPLOY_MPS_OPERATOR(Not, Not);
DEPLOY_MPS_OPERATOR(And, And);
DEPLOY_MPS_OPERATOR(Or, Or);
DEPLOY_MPS_OPERATOR(Xor, Xor);
DEPLOY_MPS_OPERATOR(Equal, Equal);
DEPLOY_MPS_OPERATOR(NotEqual, NotEqual);
DEPLOY_MPS_OPERATOR(Less, Less);
DEPLOY_MPS_OPERATOR(LessEqual, LessEqual);
DEPLOY_MPS_OPERATOR(Greater, Greater);
DEPLOY_MPS_OPERATOR(GreaterEqual, GreaterEqual);
#endif

OPERATOR_SCHEMA(Neg).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Sign).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Ceil).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Floor).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Round).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Exp).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Log).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Reciprocal).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Sqrt).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Rsqrt).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(BitwiseNot).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Abs).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(Square).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(Sin).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(Cos).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(IsInf).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(IsNaN).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(IsFinite).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(Not).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(Add).NumInputs(2).NumOutputs(1).AllowInplace({{0, 0}, {1, 0}});
OPERATOR_SCHEMA(Sub).NumInputs(2).NumOutputs(1).AllowInplace({{0, 0}, {1, 0}});
OPERATOR_SCHEMA(Mul).NumInputs(2).NumOutputs(1).AllowInplace({{0, 0}, {1, 0}});
OPERATOR_SCHEMA(Div).NumInputs(2).NumOutputs(1).AllowInplace({{0, 0}, {1, 0}});
OPERATOR_SCHEMA(Pow).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(Atan2).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(Minimum).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(Maximum).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(BitwiseAnd)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}});
OPERATOR_SCHEMA(BitwiseOr).NumInputs(2).NumOutputs(1).AllowInplace(
    {{0, 0}, {1, 0}});
OPERATOR_SCHEMA(BitwiseXor)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}});
OPERATOR_SCHEMA(And).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(Or).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(Xor).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(Equal).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(NotEqual).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(Less).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(LessEqual).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(Greater).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(GreaterEqual).NumInputs(2).NumOutputs(1);

NO_GRADIENT(Ceil);
NO_GRADIENT(Floor);
NO_GRADIENT(Round);
NO_GRADIENT(IsInf);
NO_GRADIENT(IsNaN);
NO_GRADIENT(IsFinite);
NO_GRADIENT(Atan2);
NO_GRADIENT(BitwiseNot);
NO_GRADIENT(BitwiseAnd);
NO_GRADIENT(BitwiseOr);
NO_GRADIENT(BitwiseXor);
NO_GRADIENT(Not);
NO_GRADIENT(And);
NO_GRADIENT(Or);
NO_GRADIENT(Xor);
NO_GRADIENT(Equal);
NO_GRADIENT(NotEqual);
NO_GRADIENT(Less);
NO_GRADIENT(LessEqual);
NO_GRADIENT(Greater);
NO_GRADIENT(GreaterEqual);

} // namespace dragon
