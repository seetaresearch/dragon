#include "operators/ndarray/repeat_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void RepeatOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    kernel::Repeat(Output(0)->count(),
                            outer_dim,
                                  dim,
                            inner_dim,
                            repeats(),
                                Xdata,
                                Ydata,
                              &ctx());
}

template <class Context>
void RepeatOp<Context>::RunOnDevice() {
    if (axis == -1) {
        outer_dim = inner_dim = 1;
        dim = Input(0).count();
        Output(0)->Reshape(vector<TIndex>(1, dim * repeats()));
    } else {
        outer_dim = Input(0).count(0, axis);
        dim = Input(0).dim(axis);
        inner_dim = Input(0).count(axis + 1);
        vector<TIndex> dims = Input(0).dims();
        dims[axis] *= repeats();
        Output(0)->Reshape(dims);
    }

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(Repeat);
#ifdef WITH_CUDA
DEPLOY_CUDA(Repeat);
#endif
OPERATOR_SCHEMA(Repeat).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void RepeatGradientOp<Context>::RunWithType() {
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    kernel::RepeatGrad(Output(0)->count(),
                                outer_dim,
                                      dim,
                                inner_dim,
                                repeats(),
                                   dYdata,
                                   dXdata,
                                  &ctx());
}

template <class Context>
void RepeatGradientOp<Context>::RunOnDevice() {
    if (axis == -1) {
        outer_dim = inner_dim = 1;
        dim = Input(0).count();
    } else {
        outer_dim = Input(0).count(0, axis);
        dim = Input(0).dim(axis);
        inner_dim = Input(0).count(axis + 1);
    }
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(RepeatGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(RepeatGradient);
#endif
OPERATOR_SCHEMA(RepeatGradient).NumInputs(2).NumOutputs(1);

class GetRepeatGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetRepeatGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Repeat, GetRepeatGradient);

}    // namespace dragon