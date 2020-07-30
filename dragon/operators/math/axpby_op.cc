#include "dragon/core/workspace.h"
#include "dragon/operators/math/elementwise_ops.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void AxpbyOp<Context>::DoRunWithType(Tensor* X, Tensor* Y) {
  CHECK_EQ(X->count(), Y->count());
  auto* x = X->template data<T, Context>();
  auto* y = Y->template mutable_data<T, Context>();
  if (beta_ == 1.f) {
    if (alpha_ == 1.f) {
      math::Add(X->count(), x, y, y, ctx());
    } else {
      math::Axpy(X->count(), alpha_, x, y, ctx());
    }
  } else {
    if (alpha_ == 0.f) {
      math::Scale(X->count(), beta_, y, y, ctx());
    } else {
      math::Axpby(X->count(), alpha_, x, beta_, y, ctx());
    }
  }
}

template <class Context>
void AxpbyOp<Context>::RunOnDevice() {
  for (int i = 0; i < InputSize(); i++) {
    auto &X = Input(i), *Y = Output(i);
    Y->ReshapeLike(X);
    if (XIsType(X, int8_t)) {
      DoRunWithType<int8_t>(&X, Y);
    } else if (XIsType(X, uint8_t)) {
      DoRunWithType<uint8_t>(&X, Y);
    } else if (XIsType(X, int)) {
      DoRunWithType<int>(&X, Y);
    } else if (XIsType(X, int64_t)) {
      DoRunWithType<int64_t>(&X, Y);
    } else if (XIsType(X, float16)) {
      DoRunWithType<float16>(&X, Y);
    } else if (XIsType(X, float)) {
      DoRunWithType<float>(&X, Y);
    } else if (XIsType(X, double)) {
      DoRunWithType<double>(&X, Y);
    } else
      LOG(FATAL) << MessageForUnsupported(
          types::to_string(X.meta()),
          {"int8", "uint8", "int32", "int64", "float16", "float32", "float64"});
  }
}

DEPLOY_CPU(Axpby);
#ifdef USE_CUDA
DEPLOY_CUDA(Axpby);
#endif

OPERATOR_SCHEMA(Axpby)
    /* X1, ... */
    .NumInputs(1, INT_MAX)
    /* Y1, ... */
    .NumOutputs(1, INT_MAX);

NO_GRADIENT(Axpby);

} // namespace dragon
