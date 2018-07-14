#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/recurrent/rnn_param_op.h"

namespace dragon {

template <class Context> template <typename T>
void RNNParamSetOp<Context>::RunWithType() {
    auto* Pdata = Input(0).template data<T, Context>();
    auto* Wdata = Output(0)->template mutable_data<T, Context>();
    TIndex matrix_count = 0, bias_count = 0, offset = 0, size = -1;
    for (int i = 0; i < num_layers; ++i) {
        for (int j = 0; j < num_directions; ++j) {
            TIndex pseudo_layer_id = i * num_directions + j;
            for (int k = 0; k < num_params; ++k) {
                if (layer_id == pseudo_layer_id && param_id == k)
                    size = offset = (param_type == "matrix" ? matrix_count : bias_count);
                if (k < spliter) {
                    matrix_count += (hidden_size * (i == 0 ?
                        input_size : input_ex_size));
                } else { matrix_count += hidden_size * hidden_size; }
                bias_count += hidden_size;
                if (layer_id == pseudo_layer_id && param_id == k)
                    size = (param_type == "matrix" ? matrix_count : bias_count) - size;
            }
        }
    }
    CHECK_EQ(size, Input(0).count())
        << "\nExcepted the size of param is " << size
        << ", but got " << Input(0).count();
    offset += param_type == "bias" ? matrix_count : 0;
    ctx().template Copy<T, Context, Context>(size, Wdata + offset, Pdata);
}

template <class Context>
void RNNParamSetOp<Context>::RunOnDevice() {
    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(RNNParamSet);
#ifdef WITH_CUDA
DEPLOY_CUDA(RNNParamSet);
#endif
OPERATOR_SCHEMA(RNNParamSet).NumInputs(1).NumOutputs(1);

NO_GRADIENT(RNNParamSet);

}    // namespace dragon