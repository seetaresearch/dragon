#ifdef USE_MLU

#include "dragon/operators/math/elementwise_op.h"

namespace dragon {

#define DISPATCH_WITH_TENSOR_TYPES(name, tensor_types, X_ref) \
  template <class Context>                                    \
  void name##Op<Context>::RunOnDevice() {                     \
    DispatchHelper<tensor_types>::Call(this, X_ref);          \
  }

DISPATCH_WITH_TENSOR_TYPES(CNNLIsInf, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(CNNLIsNaN, dtypes::Floating, Input(0));
DISPATCH_WITH_TENSOR_TYPES(CNNLIsFinite, dtypes::Floating, Input(0));
#undef DISPATCH_WITH_TENSOR_TYPES

template <class Context>
template <typename T>
void CNNLIsInfOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  CNNLSetTensorDesc<T>(input_desc_, X.dims());
  CNNLSetTensorDesc<bool>(output_desc_, X.dims());
  CNNL_CHECK(cnnlIsInf(
      ctx()->cnnl_handle(),
      input_desc_,
      X.template data<T, Context>(),
      CNNL_INF,
      false,
      nullptr,
      0,
      output_desc_,
      Y->ReshapeLike(X)->template mutable_data<bool, Context>()));
}

template <class Context>
template <typename T>
void CNNLIsNaNOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  CNNLSetTensorDesc<T>(input_desc_, X.dims());
  CNNLSetTensorDesc<bool>(output_desc_, X.dims());
  CNNL_CHECK(cnnlIsNan(
      ctx()->cnnl_handle(),
      input_desc_,
      X.template data<T, Context>(),
      output_desc_,
      Y->ReshapeLike(X)->template mutable_data<bool, Context>()));
}

template <class Context>
template <typename T>
void CNNLIsFiniteOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  CNNLSetTensorDesc<T>(input_desc_, X.dims());
  CNNLSetTensorDesc<bool>(output_desc_, X.dims());
  CNNL_CHECK(cnnlIsFinite(
      ctx()->cnnl_handle(),
      input_desc_,
      X.template data<T, Context>(),
      output_desc_,
      Y->ReshapeLike(X)->template mutable_data<bool, Context>()));
}

template <class Context>
template <typename T>
void CNNLNaNToNumOp<Context>::DoRunWithType() {
  CNNLSetTensorDesc<T>(input_desc_, X_->dims());
  CNNL_CHECK(cnnlNanToNum(
      ctx()->cnnl_handle(),
      input_desc_,
      X_->template data<T, Context>(),
      nan_,
      pos_inf_,
      neg_inf_,
      input_desc_,
      Y_->ReshapeLike(*X_)->template mutable_data<T, Context>()));
}

DEPLOY_CNNL_OPERATOR(IsInf);
DEPLOY_CNNL_OPERATOR(IsNaN);
DEPLOY_CNNL_OPERATOR(IsFinite);
DEPLOY_CNNL_OPERATOR(NaNToNum);

} // namespace dragon

#endif // USE_MLU
