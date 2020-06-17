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
DISPATCH_WITH_TENSOR_TYPES(Invert, IntegralTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Square, MathTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Sign, MathTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Abs, MathTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(IsInf, FloatingTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(IsNaN, FloatingTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Pow, FloatingTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Minimum, MathTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Maximum, MathTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Equal, MathTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(NotEqual, MathTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Less, MathTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(LessEqual, MathTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(Greater, MathTensorTypes, Input(0));
DISPATCH_WITH_TENSOR_TYPES(GreaterEqual, MathTensorTypes, Input(0));
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
    } else if (utils::math::IsBinaryBroadcast(A.dims(), B.dims(), Y_dims)) { \
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

DEPLOY_CPU(Ceil);
DEPLOY_CPU(Floor);
DEPLOY_CPU(Round);
DEPLOY_CPU(Sqrt);
DEPLOY_CPU(Rsqrt);
DEPLOY_CPU(Exp);
DEPLOY_CPU(Log);
DEPLOY_CPU(Sin);
DEPLOY_CPU(Cos);
DEPLOY_CPU(Invert);
DEPLOY_CPU(Square);
DEPLOY_CPU(Sign);
DEPLOY_CPU(Abs);
DEPLOY_CPU(IsInf);
DEPLOY_CPU(IsNaN);
DEPLOY_CPU(Pow);
DEPLOY_CPU(Minimum);
DEPLOY_CPU(Maximum);
DEPLOY_CPU(Equal);
DEPLOY_CPU(NotEqual);
DEPLOY_CPU(Less);
DEPLOY_CPU(LessEqual);
DEPLOY_CPU(Greater);
DEPLOY_CPU(GreaterEqual);

#ifdef USE_CUDA
DEPLOY_CUDA(Ceil);
DEPLOY_CUDA(Floor);
DEPLOY_CUDA(Round);
DEPLOY_CUDA(Sqrt);
DEPLOY_CUDA(Rsqrt);
DEPLOY_CUDA(Exp);
DEPLOY_CUDA(Log);
DEPLOY_CUDA(Sin);
DEPLOY_CUDA(Cos);
DEPLOY_CUDA(Invert);
DEPLOY_CUDA(Square);
DEPLOY_CUDA(Sign);
DEPLOY_CUDA(Abs);
DEPLOY_CUDA(IsInf);
DEPLOY_CUDA(IsNaN);
DEPLOY_CUDA(Pow);
DEPLOY_CUDA(Minimum);
DEPLOY_CUDA(Maximum);
DEPLOY_CUDA(Equal);
DEPLOY_CUDA(NotEqual);
DEPLOY_CUDA(Less);
DEPLOY_CUDA(LessEqual);
DEPLOY_CUDA(Greater);
DEPLOY_CUDA(GreaterEqual);
#endif

OPERATOR_SCHEMA(Ceil).NumInputs(1).NumOutputs(1).Inplace({{0, 0}});
OPERATOR_SCHEMA(Floor).NumInputs(1).NumOutputs(1).Inplace({{0, 0}});
OPERATOR_SCHEMA(Round).NumInputs(1).NumOutputs(1).Inplace({{0, 0}});
OPERATOR_SCHEMA(Sign).NumInputs(1).NumOutputs(1).Inplace({{0, 0}});
OPERATOR_SCHEMA(Sqrt).NumInputs(1).NumOutputs(1).Inplace({{0, 0}});
OPERATOR_SCHEMA(Rsqrt).NumInputs(1).NumOutputs(1).Inplace({{0, 0}});
OPERATOR_SCHEMA(Exp).NumInputs(1).NumOutputs(1).Inplace({{0, 0}});
OPERATOR_SCHEMA(Log).NumInputs(1).NumOutputs(1).Inplace({{0, 0}});
OPERATOR_SCHEMA(Invert).NumInputs(1).NumOutputs(1).Inplace({{0, 0}});
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
