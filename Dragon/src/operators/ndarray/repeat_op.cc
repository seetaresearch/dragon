#include "operators/ndarray/repeat_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void RepeatOp<Context>::RunWithType() {
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    kernel::Repeat(output(0)->count(),
                            outer_dim,
                                  dim,
                            inner_dim,
                                 reps,
                                Xdata,
                                Ydata,
                              &ctx());
}

template <class Context>
void RepeatOp<Context>::RunOnDevice() {
    //  parse repeats from desc
    Tensor* repeats = ws()->GetTensor(repeats_desc);
    CHECK(repeats->IsType<int>()) << "\nThe type of repeats should be int32.";
    reps = repeats->template data<int, CPUContext>()[0];
    if (axis == -1) {
        outer_dim = inner_dim = 1;
        dim = input(0).count();
        output(0)->Reshape(vector<TIndex>(1, dim * reps));
    } else {
        outer_dim = input(0).count(0, axis);
        dim = input(0).dim(axis);
        inner_dim = input(0).count(axis + 1);
        vector<TIndex> dims = input(0).dims();
        dims[axis] *= reps;
        output(0)->Reshape(dims);
    }

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(Repeat);
#ifdef WITH_CUDA
DEPLOY_CUDA(Repeat);
#endif
OPERATOR_SCHEMA(Repeat).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void RepeatGradientOp<Context>::RunWithType() {
    auto* dYdata = input(-1).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    kernel::RepeatGrad(output(0)->count(),
                                outer_dim,
                                      dim,
                                inner_dim,
                                     reps,
                                   dYdata,
                                   dXdata,
                                  &ctx());
}

template <class Context>
void RepeatGradientOp<Context>::RunOnDevice() {
    //  parse repeats from desc
    Tensor* repeats = ws()->GetTensor(repeats_desc);
    CHECK(repeats->IsType<int>()) << "\nThe type of repeats should be int32.";
    reps = repeats->template data<int, CPUContext>()[0];
    if (axis == -1) {
        outer_dim = inner_dim = 1;
        dim = input(0).count();
    } else {
        outer_dim = input(0).count(0, axis);
        dim = input(0).dim(axis);
        inner_dim = input(0).count(axis + 1);
    }
    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
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
