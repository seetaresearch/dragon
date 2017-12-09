#include "operators/ndarray/random_pick_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void RandomPickOp<Context>::RunWithType() {
    auto* indices = pick_indices->template mutable_data<T, CPUContext>();
    for (int i = 0; i < pick_indices->count(); i++)
        indices[i] = T((*rand_generator())() % x_slice_dim);

    auto* Xdata = input(0).template data<T, Context>();
    indices = pick_indices->template mutable_data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    kernel::At<T, Context>(output(0)->count(), outer_dim, inner_dim,
                                                        x_slice_dim,
                                                        y_slice_dim,
                                                            indices,
                                                              Xdata,
                                                              Ydata,
                                                            &ctx());
}

template <class Context>
void RandomPickOp<Context>::RunOnDevice() {
    output_dims = input(0).dims();
    x_slice_dim = input(0).dim(axis);
    output_dims[axis] = y_slice_dim = max_samples;

    outer_dim = input(0).count(0, axis);
    inner_dim = input(0).count(axis + 1);
    output(0)->Reshape(output_dims);

    pick_indices = ws()->CreateTensor("/mnt/" + anchor() + "/pick_indices");
    pick_indices->Reshape(vector<TIndex>(1, max_samples));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";

    if (output(1)->name() != "ignore") {
        output(1)->ReshapeLike(*pick_indices);
        output(1)->Share(*pick_indices);
    }
}

DEPLOY_CPU(RandomPick);
#ifdef WITH_CUDA
DEPLOY_CUDA(RandomPick);
#endif
OPERATOR_SCHEMA(RandomPick).NumInputs(1).NumOutputs(2);

template <class Context> template <typename T>
void RandomPickGradientOp<Context>::RunWithType() {
    auto* indices = pick_indices->template data<T, Context>();
    auto* dYdata = input(-1).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    math::Set<T, Context>(output(0)->count(), 0, dXdata);
    kernel::AtGrad<T, Context>(input(-1).count(), outer_dim, inner_dim,
                                                           x_slice_dim, 
                                                           y_slice_dim, 
                                                               indices, 
                                                                dYdata, 
                                                                dXdata, 
                                                               &ctx());
}

template <class Context>
void RandomPickGradientOp<Context>::RunOnDevice() {
    pick_indices = ws()->GetTensor("/mnt/" + anchor() + "/pick_indices");

    x_slice_dim = input(0).dim(axis);
    y_slice_dim = pick_indices->count();
    outer_dim = input(0).count(0, axis);
    inner_dim = input(0).count(axis + 1);
    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
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