#include "dragon/operators/math/elementwise_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void AxpbyOp<Context>::DoRunWithType() {
  auto* x = X_->template data<T, Context>();
  auto* y = Y_->ReshapeLike(*X_)->template mutable_data<T, Context>();
  if (beta_ == 1.f) {
    if (alpha_ == 1.f) {
      math::Add(X_->count(), x, y, y, ctx());
    } else {
      math::Axpy(X_->count(), alpha_, x, y, ctx());
    }
  } else {
    if (alpha_ == 0.f) {
      math::Scale(X_->count(), beta_, y, y, ctx());
    } else {
      math::Axpby(X_->count(), alpha_, x, beta_, y, ctx());
    }
  }
}

template <class Context>
void AxpbyOp<Context>::RunOnDevice() {
  for (int i = 0; i < InputSize(); ++i) {
    X_ = &Input(i), Y_ = Output(i);
    DispatchHelper<dtypes::Numerical>::Call(this, *X_);
  }
}

DEPLOY_CPU_OPERATOR(Axpby);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Axpby);
#endif

OPERATOR_SCHEMA(Axpby).AllowInplace([](int, int) -> bool { return true; });

NO_GRADIENT(Axpby);

} // namespace dragon
