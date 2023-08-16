#include "dragon/operators/math/cast_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename InputT, typename OutputT>
bool CastOp<Context>::MaybeConvert() {
  auto &X = Input(0), *Y = Output(0);
  if (dtypes::to_string(X.meta()) == data_type()) {
    Y->ReshapeLike(X)->CopyFrom(X, ctx());
    return true;
  }
  if (dtypes::to_string<OutputT>() == data_type()) {
    const auto N = X.count();
    if ((void*)&X != (void*)Y) {
      math::Cast(
          N,
          X.template data<InputT, Context>(),
          Y->ReshapeLike(X)->template mutable_data<OutputT, Context>(),
          ctx());
    } else {
      auto* scratch = ctx()->workspace()->template data<OutputT, Context>(N);
      math::Cast(N, X.template data<InputT, Context>(), scratch, ctx());
      math::Copy(
          N,
          scratch,
          Y->ReshapeLike(X)->template mutable_data<OutputT, Context>(),
          ctx());
    }
    return true;
  }
  return false;
}

template <class Context>
template <typename T>
void CastOp<Context>::DoRunWithType() {
  if (MaybeConvert<T, bool>()) {
  } else if (MaybeConvert<T, uint8_t>()) {
  } else if (MaybeConvert<T, int8_t>()) {
  } else if (MaybeConvert<T, int>()) {
  } else if (MaybeConvert<T, int64_t>()) {
  } else if (MaybeConvert<T, float16>()) {
  } else if (MaybeConvert<T, bfloat16>()) {
  } else if (MaybeConvert<T, float>()) {
  } else if (MaybeConvert<T, double>()) {
  } else {
    LOG(FATAL) << MessageForUnsupported(
        data_type(),
        {"bool",
         "uint8",
         "int8",
         "int32",
         "int64",
         "float16",
         "bfloat16",
         "float32",
         "float64"});
  }
}

template <class Context>
void CastOp<Context>::RunOnDevice() {
  auto& X = Input(0);
  Output("X_spec")->set_meta(X.meta());
  DispatchHelper<dtypes::TypesBase<
      bool,
      uint8_t,
      int8_t,
      int,
      int64_t,
      float16,
      bfloat16,
      float,
      double>>::Call(this, X);
}

template <class Context>
void CastGradientOp<Context>::RunOnDevice() {
  this->data_type_ = dtypes::to_string(Input("X_spec").meta());
  DispatchHelper<dtypes::TypesBase<
      bool,
      uint8_t,
      int8_t,
      int,
      int64_t,
      float16,
      bfloat16,
      float,
      double>>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Cast);
DEPLOY_CPU_OPERATOR(CastGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Cast);
DEPLOY_CUDA_OPERATOR(CastGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(Cast, Cast);
DEPLOY_MPS_OPERATOR(CastGradient, CastGradient);
#endif
#ifdef USE_MLU
DEPLOY_MLU_OPERATOR(Cast);
DEPLOY_MLU_OPERATOR(CastGradient);
#endif

OPERATOR_SCHEMA(Cast)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(CastGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{0, 0}});

REGISTER_GRADIENT(Cast, SimpleGradientMaker);

} // namespace dragon
