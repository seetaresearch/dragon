#include "dragon/operators/math/elementwise_ops.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

#define DISPATCH_WITH_TENSOR_TYPES(name, tensor_types, X_ref) \
  template <class Context>                                    \
  void name##Op<Context>::RunOnDevice() {                     \
    DispatchHelper<tensor_types>::Call(this, X_ref);          \
  }

DISPATCH_WITH_TENSOR_TYPES(Ceil, FloatingTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Floor, FloatingTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Round, FloatingTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Sqrt, FloatingTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Rsqrt, FloatingTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Exp, FloatingTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Log, FloatingTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Sin, FloatingTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Cos, FloatingTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Invert, BooleanIntegralTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Square, NumericalTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Sign, NumericalTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Abs, NumericalTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(IsInf, FloatingTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(IsNaN, FloatingTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Pow, FloatingTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Minimum, NumericalTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Maximum, NumericalTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Equal, NumericalTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(NotEqual, NumericalTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Less, NumericalTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(LessEqual, NumericalTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Greater, NumericalTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(GreaterEqual, NumericalTensorTypes, Input(0));
#undef DISPATCH_WITH_TENSOR_TYPES

#define DEFINE_SIMPLE_UNARY_OP_IMPL(name, TOut)                    \
  template <class Context>                                         \
  template <typename T>                                            \
  void name##Op<Context>::DoRunWithType() {                        \
    auto &X = Input(0), *Y = Output(0);                            \
    math::name(                                                    \
        X.count(),                                                 \
        X.template data<T, Context>(),                             \
        Y->ReshapeLike(X)->template mutable_data<TOut, Context>(), \
        ctx());                                                    \
  }

DEFINE_SIMPLE_UNARY_OP_IMPL(Sin, T);
DEFINE_SIMPLE_UNARY_OP_IMPL(Cos, T);
DEFINE_SIMPLE_UNARY_OP_IMPL(Square, T);
DEFINE_SIMPLE_UNARY_OP_IMPL(Abs, T);
DEFINE_SIMPLE_UNARY_OP_IMPL(IsInf, bool);
DEFINE_SIMPLE_UNARY_OP_IMPL(IsNaN, bool);
#undef DEFINE_SIMPLE_UNARY_OP_IMPL

#define DEFINE_INPLACE_UNARY_OP_IMPL(name, TOut)                   \
  template <class Context>                                         \
  template <typename T>                                            \
  void name##Op<Context>::DoRunWithType() {                        \
    auto &X = Input(0), *Y = Output(0, {0});                       \
    math::name(                                                    \
        X.count(),                                                 \
        X.template data<T, Context>(),                             \
        Y->ReshapeLike(X)->template mutable_data<TOut, Context>(), \
        ctx());                                                    \
  }

DEFINE_INPLACE_UNARY_OP_IMPL(Ceil, T);
DEFINE_INPLACE_UNARY_OP_IMPL(Floor, T);
DEFINE_INPLACE_UNARY_OP_IMPL(Round, T);
DEFINE_INPLACE_UNARY_OP_IMPL(Sign, T);
DEFINE_INPLACE_UNARY_OP_IMPL(Sqrt, T);
DEFINE_INPLACE_UNARY_OP_IMPL(Rsqrt, T);
DEFINE_INPLACE_UNARY_OP_IMPL(Exp, T);
DEFINE_INPLACE_UNARY_OP_IMPL(Log, T);
DEFINE_INPLACE_UNARY_OP_IMPL(Invert, T);
#undef DEFINE_INPLACE_UNARY_OP_IMPL

#define DEFINE_SIMPLE_BINARY_OP_IMPL(name, TOut)                             \
  template <class Context>                                                   \
  template <typename T>                                                      \
  void name##Op<Context>::DoRunWithType() {                                  \
    auto &A = Input(0), &B = Input(1), *Y = Output(0);                       \
                                                                             \
    vec64_t Y_dims(A.dims());                                                \
    if (A.dims() == B.dims()) {                                              \
      math::name(                                                            \
          A.count(),                                                         \
          A.template data<T, Context>(),                                     \
          B.template data<T, Context>(),                                     \
          Y->Reshape(Y_dims)->template mutable_data<TOut, Context>(),        \
          ctx());                                                            \
    } else if (math::utils::IsBinaryBroadcast(A.dims(), B.dims(), Y_dims)) { \
      math::name(                                                            \
          A.ndim(),                                                          \
          A.dims().data(),                                                   \
          B.ndim(),                                                          \
          B.dims().data(),                                                   \
          A.template data<T, Context>(),                                     \
          B.template data<T, Context>(),                                     \
          Y->Reshape(Y_dims)->template mutable_data<TOut, Context>(),        \
          ctx());                                                            \
    } else {                                                                 \
      LOG(FATAL) << "Could not broadcast together with shapes: "             \
                 << A.DimString() << " " << B.DimString();                   \
    }                                                                        \
  }

DEFINE_SIMPLE_BINARY_OP_IMPL(Pow, T);
DEFINE_SIMPLE_BINARY_OP_IMPL(Minimum, T);
DEFINE_SIMPLE_BINARY_OP_IMPL(Maximum, T);
DEFINE_SIMPLE_BINARY_OP_IMPL(Equal, bool);
DEFINE_SIMPLE_BINARY_OP_IMPL(NotEqual, bool);
DEFINE_SIMPLE_BINARY_OP_IMPL(Less, bool);
DEFINE_SIMPLE_BINARY_OP_IMPL(LessEqual, bool);
DEFINE_SIMPLE_BINARY_OP_IMPL(Greater, bool);
DEFINE_SIMPLE_BINARY_OP_IMPL(GreaterEqual, bool);
#undef DEFINE_SIMPLE_BINARY_OP_IMPL

DEPLOY_CPU_OPERATOR(Ceil);
DEPLOY_CPU_OPERATOR(Floor);
DEPLOY_CPU_OPERATOR(Round);
DEPLOY_CPU_OPERATOR(Sqrt);
DEPLOY_CPU_OPERATOR(Rsqrt);
DEPLOY_CPU_OPERATOR(Exp);
DEPLOY_CPU_OPERATOR(Log);
DEPLOY_CPU_OPERATOR(Sin);
DEPLOY_CPU_OPERATOR(Cos);
DEPLOY_CPU_OPERATOR(Invert);
DEPLOY_CPU_OPERATOR(Square);
DEPLOY_CPU_OPERATOR(Sign);
DEPLOY_CPU_OPERATOR(Abs);
DEPLOY_CPU_OPERATOR(IsInf);
DEPLOY_CPU_OPERATOR(IsNaN);
DEPLOY_CPU_OPERATOR(Pow);
DEPLOY_CPU_OPERATOR(Minimum);
DEPLOY_CPU_OPERATOR(Maximum);
DEPLOY_CPU_OPERATOR(Equal);
DEPLOY_CPU_OPERATOR(NotEqual);
DEPLOY_CPU_OPERATOR(Less);
DEPLOY_CPU_OPERATOR(LessEqual);
DEPLOY_CPU_OPERATOR(Greater);
DEPLOY_CPU_OPERATOR(GreaterEqual);

#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Ceil);
DEPLOY_CUDA_OPERATOR(Floor);
DEPLOY_CUDA_OPERATOR(Round);
DEPLOY_CUDA_OPERATOR(Sqrt);
DEPLOY_CUDA_OPERATOR(Rsqrt);
DEPLOY_CUDA_OPERATOR(Exp);
DEPLOY_CUDA_OPERATOR(Log);
DEPLOY_CUDA_OPERATOR(Sin);
DEPLOY_CUDA_OPERATOR(Cos);
DEPLOY_CUDA_OPERATOR(Invert);
DEPLOY_CUDA_OPERATOR(Square);
DEPLOY_CUDA_OPERATOR(Sign);
DEPLOY_CUDA_OPERATOR(Abs);
DEPLOY_CUDA_OPERATOR(IsInf);
DEPLOY_CUDA_OPERATOR(IsNaN);
DEPLOY_CUDA_OPERATOR(Pow);
DEPLOY_CUDA_OPERATOR(Minimum);
DEPLOY_CUDA_OPERATOR(Maximum);
DEPLOY_CUDA_OPERATOR(Equal);
DEPLOY_CUDA_OPERATOR(NotEqual);
DEPLOY_CUDA_OPERATOR(Less);
DEPLOY_CUDA_OPERATOR(LessEqual);
DEPLOY_CUDA_OPERATOR(Greater);
DEPLOY_CUDA_OPERATOR(GreaterEqual);
#endif

OPERATOR_SCHEMA(Ceil).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Floor).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Round).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Sign).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Sqrt).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Rsqrt).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Exp).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Log).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Invert).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
OPERATOR_SCHEMA(Sin).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(Cos).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(Square).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(Abs).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(IsInf).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(IsNaN).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(Pow).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(Minimum).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(Maximum).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(Equal).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(NotEqual).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(Less).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(LessEqual).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(Greater).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(GreaterEqual).NumInputs(2).NumOutputs(1);

NO_GRADIENT(Ceil);
NO_GRADIENT(Floor);
NO_GRADIENT(Round);
NO_GRADIENT(Invert);
NO_GRADIENT(IsInf);
NO_GRADIENT(IsNaN);
NO_GRADIENT(Equal);
NO_GRADIENT(NotEqual);
NO_GRADIENT(Less);
NO_GRADIENT(LessEqual);
NO_GRADIENT(Greater);
NO_GRADIENT(GreaterEqual);

} // namespace dragon
