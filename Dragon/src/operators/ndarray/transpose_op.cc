#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/ndarray/transpose_op.h"

namespace dragon {

template <class Context> template <typename T>
void TransposeOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* OSdata = old_steps->template data<int, Context>();
    auto* NSdata = new_steps->template data<int, Context>();
    auto* ORdata = order->template data <int, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::Transpose<T, Context>(
        Output(0)->count(), (int)Output(0)->ndim(),
            ORdata, OSdata, NSdata, Xdata, Ydata, ctx());
}

template <class Context>
void TransposeOp<Context>::RunOnDevice() {
    auto given_n_perms = std::max(
        perms_desc.size(), perms_value.size());
    if (given_n_perms == 0) {
        // Reverse dimensions directly if missing perms
        perms_value.clear(); given_n_perms = Input(0).ndim();
        for (int i = (int)given_n_perms - 1; i >= 0; i--)
            perms_value.push_back(i);
    }
    CHECK_EQ(Input(0).ndim(), given_n_perms)
        << "\nProvide " << given_n_perms << " dims to permsute, "
        << "but Tensor(" << Input(0).name() << ")'s dims are "
        << Input(0).DimString();
   
    vector<TIndex> output_dims;
    order = ws()->CreateTensor("/mnt/" + anchor() + "/transpose/order");
    old_steps = ws()->CreateTensor("/mnt/" + anchor() + "/transpose/old_steps");
    new_steps = ws()->CreateTensor("/mnt/" + anchor() + "/transpose/new_steps");
    order->Reshape({ (TIndex)given_n_perms });
    old_steps->Reshape({ (TIndex)given_n_perms });
    new_steps->Reshape({ (TIndex)given_n_perms });
    auto* OSdata = old_steps->template mutable_data<int, CPUContext>();
    auto* NSdata = new_steps->template mutable_data<int, CPUContext>();
    auto* ORdata = order->template mutable_data<int, CPUContext>();
    for (int i = 0; i < given_n_perms; i++) {
        if (i == given_n_perms - 1) OSdata[i] = 1;
        else OSdata[i] = Input(0).count(i + 1);
        ORdata[i] = perms(i);
        output_dims.push_back(Input(0).dim(perms(i)));
    }
    Output(0)->Reshape(output_dims);
    for (int i = 0; i < given_n_perms; i++) {
        if (i == given_n_perms - 1) NSdata[i] = 1;
        else NSdata[i] = Output(0)->count(i + 1);
    }

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(Transpose);
#ifdef WITH_CUDA
DEPLOY_CUDA(Transpose);
#endif
OPERATOR_SCHEMA(Transpose).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void TransposeGradientOp<Context>::RunWithType() {
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* OSdata = old_steps->template data<int, Context>();
    auto* NSdata = new_steps->template data<int, Context>();
    auto* ORdata = order->template data <int, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    kernel::TransposeGrad<T, Context>(
        Input(-1).count(), order->count(),
            ORdata, OSdata, NSdata, dYdata, dXdata, ctx());
}

template <class Context>
void TransposeGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));
    order = ws()->GetTensor("/mnt/" + anchor() + "/transpose/order");
    old_steps = ws()->GetTensor("/mnt/" + anchor() + "/transpose/old_steps");
    new_steps = ws()->GetTensor("/mnt/" + anchor() + "/transpose/new_steps");

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}
    
DEPLOY_CPU(TransposeGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(TransposeGradient);
#endif
OPERATOR_SCHEMA(TransposeGradient).NumInputs(2).NumOutputs(1);

class GetTransposeGradient final : public GradientMakerBase{
 public:
    GRADIENT_MAKER_CTOR(GetTransposeGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Transpose, GetTransposeGradient);

}  // namespace dragon