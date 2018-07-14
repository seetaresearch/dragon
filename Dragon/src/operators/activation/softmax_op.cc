#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"
#include "operators/activation/softmax_op.h"

namespace dragon {

template <class Context> template <typename T>
void SoftmaxOp<Context>::RunWithType() {
    DECLARE_MULTIPLIER(multiplier, Input(0).dim(axis));
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    auto* WSdata = ws()->template caches<T, Context>({ Input(0).count() })[0];

    ctx().template Copy<T, Context, Context>(
        Input(0).count(), Ydata, Xdata);

    kernel::Softmax<T, Context>(
        Output(0)->count(), Input(0).dim(axis),
            outer_dim, inner_dim, multiplier,
                Xdata, WSdata, Ydata, &ctx());
}

template <class Context>
void SoftmaxOp<Context>::RunOnDevice() {
    if (axis == -1) axis = (int)Input(0).ndim() - 1;
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(Softmax);
#ifdef WITH_CUDA
DEPLOY_CUDA(Softmax);
#endif
OPERATOR_SCHEMA(Softmax).NumInputs(1).NumOutputs(1).Inplace({ { 0, 0 } });

template <class Context> template <typename T>
void SoftmaxGradientOp<Context>::RunWithType() {
    DECLARE_MULTIPLIER(multiplier, Input(0).dim(axis));
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* Ydata = Input(0).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* WSdata = ws()->template caches<T, Context>({ Input(0).count() })[0];

    ctx().template Copy<T, Context, Context>(
        Input(0).count(), dXdata, dYdata);

    kernel::SoftmaxGrad<T, Context>(
        Output(0)->count(), Input(0).dim(axis),
            outer_dim, inner_dim, multiplier,
                dYdata, Ydata, WSdata, dXdata, &ctx());
}

template <class Context>
void SoftmaxGradientOp<Context>::RunOnDevice() {
    if (axis == -1) axis = (int)Input(0).ndim() - 1;
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
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