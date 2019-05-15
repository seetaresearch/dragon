#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/recurrent/rnn_param_op.h"

namespace dragon {

template <class Context> template <typename T>
void RNNParamSetOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();
    int64_t ofs = 0, size = -1;
    int64_t w_count = 0, b_count = 0;
    for (int i = 0; i < nlayers_; ++i) {
        for (int j = 0; j < ndirections_; ++j) {
            int64_t pseudo_id = i * ndirections_ + j;
            for (int k = 0; k < nparams_; ++k) {
                if (layer_id_ == pseudo_id &&
                        param_id_ == k)
                    size = ofs = (
                        param_type_ == "matrix" ?
                            w_count : b_count
                    );
                if (k < spliter_) {
                    w_count += (
                        hidden_size_ * (
                            i == 0 ? 
                            input_size_ :
                            input_ex_size_
                        )
                    );
                } else {
                    w_count += hidden_size_ * hidden_size_;
                }
                b_count += hidden_size_;
                if (layer_id_ == pseudo_id &&
                        param_id_ == k)
                    size = (
                        param_type_ == "matrix" ?
                            w_count : b_count
                    ) - size;
            }
        }
    }
    CHECK_EQ(size, X(0).count())
        << "\nExcepted the size of param is " << size
        << ", but got " << X(0).count();
    ofs += param_type_ == "bias" ? w_count : 0;
    math::Copy(size, x, y + ofs, ctx());
    ctx()->FinishDeviceCompution();
}

template <class Context>
void RNNParamSetOp<Context>::RunOnDevice() {
    DispatchHelper<TensorTypes
        <float, float16>>::Call(this, X(0));
}

DEPLOY_CPU(RNNParamSet);
#ifdef WITH_CUDA
DEPLOY_CUDA(RNNParamSet);
#endif

OPERATOR_SCHEMA(RNNParamSet)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

NO_GRADIENT(RNNParamSet);

}  // namespace dragon