#ifdef USE_MLU

#include "dragon/operators/array/reverse_op.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLReverseOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  int num_dims = X.ndim();
  for (int i = 0; i < axes_.size(); ++i) {
    int axis = axes_[i];
    axis = axis < 0 ? axis + num_dims : axis;
    CHECK(axis >= 0 && axis < num_dims)
        << "\nExcepted the <axis> in [-" << num_dims << ", " << num_dims
        << "), got " << axes_[i] << ".";
  }

  CNNLSetTensorDesc<T>(input_desc_, X.dims());
  CNNL_CHECK(cnnlFlip(
      ctx()->cnnl_handle(),
      vec32_t({axes_.begin(), axes_.end()}).data(),
      axes_.size(),
      input_desc_,
      X.template data<T, Context>(),
      input_desc_,
      Y->ReshapeLike(X)->template mutable_data<T, Context>()));
}

DEPLOY_CNNL_OPERATOR(Reverse);
REGISTER_CNNL_OPERATOR(ReverseGradient, CNNLReverseOp<MLUContext>);

} // namespace dragon

#endif // USE_MLU
