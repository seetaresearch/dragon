#include "dragon/operators/recurrent/rnn_param_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void RNNParamSetOp<Context>::DoRunWithType() {
  auto* x = Input(0).template data<T, Context>();
  auto* y = Output(0)->template mutable_data<T, Context>();
  int64_t offset = 0, size = -1;
  int64_t w_count = 0, b_count = 0;
  for (int i = 0; i < nlayers_; ++i) {
    for (int j = 0; j < ndirections_; ++j) {
      int64_t pseudo_id = i * ndirections_ + j;
      for (int k = 0; k < nparams_; ++k) {
        if (layer_id_ == pseudo_id && param_id_ == k)
          size = offset = (param_type_ == "matrix" ? w_count : b_count);
        if (k < spliter_) {
          w_count += (hidden_size_ * (i == 0 ? input_size_ : input_ex_size_));
        } else {
          w_count += hidden_size_ * hidden_size_;
        }
        b_count += hidden_size_;
        if (layer_id_ == pseudo_id && param_id_ == k)
          size = (param_type_ == "matrix" ? w_count : b_count) - size;
      }
    }
  }
  CHECK_EQ(size, Input(0).count()) << "\nExcepted the size of param is " << size
                                   << ", but got " << Input(0).count();
  offset += param_type_ == "bias" ? w_count : 0;
  math::Copy(size, x, y + offset, ctx());
  ctx()->FinishDeviceComputation();
}

template <class Context>
void RNNParamSetOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(RNNParamSet);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(RNNParamSet);
#endif

OPERATOR_SCHEMA(RNNParamSet)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

NO_GRADIENT(RNNParamSet);

} // namespace dragon
