#include "operators/activation/softmax_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void SoftmaxOp<Context>::RunWithType() {
    INIT_MULTIPLIER(sum_multiplier, input(0).dim(axis));
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    auto* Sdata = scale->template mutable_data<T, Context>();
    auto* SMul_data = sum_multiplier->template data<T, Context>();

    ctx().template Copy<T, Context, Context>(input(0).count(), Ydata, Xdata);
    kernel::Softmax<T, Context>(output(0)->count(), input(0).dim(axis),
        outer_dim, inner_dim, SMul_data, Xdata, Sdata, Ydata, &ctx());
}

template <class Context>
void SoftmaxOp<Context>::RunOnDevice() {
    if (axis == -1) axis = (int)input(0).ndim() - 1;
    scale = ws()->CreateTensor("_t_softmax_scale");
    scale->ReshapeLike(input(0));
    outer_dim = input(0).count(0, axis);
    inner_dim = input(0).count(axis + 1);
    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(Softmax);
#ifdef WITH_CUDA
DEPLOY_CUDA(Softmax);
#endif
OPERATOR_SCHEMA(Softmax).NumInputs(1).NumOutputs(1).Inplace({ { 0, 0 } });

template <class Context> template <typename T>
void SoftmaxGradientOp<Context>::RunWithType() {
    INIT_MULTIPLIER(this->sum_multiplier, input(0).dim(axis));
    auto* dYdata = input(-1).template data<T, Context>();
    auto* Ydata = input(0).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    auto* Sdata = scale->template mutable_data<T, Context>();
    auto* SMul_data = sum_multiplier->template data<T, Context>();

    ctx().template Copy<T, Context, Context>(input(0).count(), dXdata, dYdata);
    kernel::SoftmaxGrad<T, Context>(output(0)->count(), input(0).dim(axis),
        outer_dim, inner_dim, SMul_data, dYdata, Ydata, Sdata, dXdata);
}

template <class Context>
void SoftmaxGradientOp<Context>::RunOnDevice() {
    if (axis == -1) axis = (int)input(0).ndim() - 1;
    scale = ws()->CreateTensor("_t_softmax_scale");
    scale->ReshapeLike(input(0));
    outer_dim = input(0).count(0, axis);
    inner_dim = input(0).count(axis + 1);
    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(SoftmaxGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SoftmaxGradient);
#endif
OPERATOR_SCHEMA(SoftmaxGradient).NumInputs(2).NumOutputs(1).Inplace({ { 1, 0 } });

class GetSoftmaxGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetSoftmaxGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {O(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Softmax, GetSoftmaxGradient);

}    // namespace dragon