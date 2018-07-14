#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/ndarray/random_pick_op.h"

namespace dragon {

template <class Context> template <typename T>
void RandomPickOp<Context>::RunWithType() {
    auto* indices = pick_indices->template mutable_data<int, CPUContext>();
    for (int i = 0; i < pick_indices->count(); i++)
        indices[i] = int((*ctx().rand_generator())() % x_slice_dim);

    auto* Xdata = Input(0).template data<T, Context>();
    indices = pick_indices->template mutable_data<int, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::Gather<T, Context>(
        Output(0)->count(), outer_dim, inner_dim,
            x_slice_dim, y_slice_dim, indices, Xdata, Ydata);
}

template <class Context>
void RandomPickOp<Context>::RunOnDevice() {
    output_dims = Input(0).dims();
    x_slice_dim = Input(0).dim(axis);
    output_dims[axis] = y_slice_dim = max_samples;

    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    Output(0)->Reshape(output_dims);

    pick_indices = ws()->CreateTensor(
        "/mnt/" + anchor() + "/pick/indices");
    pick_indices->Reshape({ max_samples });

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });

    if (Output(1)->name() != "ignore") {
        Output(1)->ReshapeLike(*pick_indices);
        Output(1)->template Copy<Context, Context>(*pick_indices);
    }
}

DEPLOY_CPU(RandomPick);
#ifdef WITH_CUDA
DEPLOY_CUDA(RandomPick);
#endif
OPERATOR_SCHEMA(RandomPick).NumInputs(1).NumOutputs(2);

template <class Context> template <typename T>
void RandomPickGradientOp<Context>::RunWithType() {
    auto* indices = pick_indices->template data<int, Context>();
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    math::Set<T, Context>(Output(0)->count(), 0, dXdata);

    kernel::GatherGrad<T, Context>(
        Input(-1).count(), outer_dim, inner_dim,
            x_slice_dim, y_slice_dim, indices, dYdata, dXdata);
}

template <class Context>
void RandomPickGradientOp<Context>::RunOnDevice() {
    pick_indices = ws()->GetTensor(
        "/mnt/" + anchor() + "/pick/indices");

    x_slice_dim = Input(0).dim(axis);
    y_slice_dim = pick_indices->count();
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(RandomPickGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(RandomPickGradient);
#endif
OPERATOR_SCHEMA(RandomPickGradient).NumInputs(2).NumOutputs(1);

class GetRandomPickGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetRandomPickGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "", 
            vector<string> {I(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(RandomPick, GetRandomPickGradient);

}   // namespace dragon