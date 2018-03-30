#include "operators/ndarray/transpose_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void TransposeOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* OSdata = old_steps->template data<int, Context>();
    auto* NSdata = new_steps->template data<int, Context>();
    auto* ORdata = order->template data <int, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    kernel::Transpose<T, Context>(Output(0)->count(), int(perms.size()),
                                  ORdata, OSdata, NSdata, 
                                  Xdata, Ydata);
}

template <class Context>
void TransposeOp<Context>::RunOnDevice() {
    if (reverse_dims) {
        perms.clear();
        for (int i = (int)Input(0).ndim() - 1; i >= 0; i--) perms.push_back(i);
    }
    CHECK_EQ(Input(0).ndim(), perms.size())
        << "\nProvide " << perms.size() << " dims to permsute,"
        << "\nbut Tensor(" << Input(0).name() << ")'s dims are "
        << Input(0).dim_string();
    vector<TIndex> output_dims;
    order = ws()->CreateTensor("/mnt/" + Anchor() + "/transpose/order");
    old_steps = ws()->CreateTensor("/mnt/" + Anchor() + "/transpose/old_steps");
    new_steps = ws()->CreateTensor("/mnt/" + Anchor() + "/transpose/new_steps");
    order->Reshape(vector<TIndex>(1, perms.size()));
    old_steps->Reshape(vector<TIndex>(1, perms.size()));
    new_steps->Reshape(vector<TIndex>(1, perms.size()));
    auto* OSdata = old_steps->template mutable_data<int, CPUContext>();
    auto* NSdata = new_steps->template mutable_data<int, CPUContext>();
    auto* ORdata = order->template mutable_data<int, CPUContext>();
    for (int i = 0; i < perms.size(); i++) {
        if (i == perms.size() - 1) OSdata[i] = 1;
        else OSdata[i] = Input(0).count(i + 1);
        ORdata[i] = perms[i];
        output_dims.push_back(Input(0).dim(perms[i]));
    }
    Output(0)->Reshape(output_dims);
    for (int i = 0; i < perms.size(); i++) {
        if (i == perms.size() - 1) NSdata[i] = 1;
        else NSdata[i] = Output(0)->count(i + 1);
    }

    if (Input(0).template IsType<float>()) RunWithType<float>();
#ifdef WITH_CUDA_FP16
    else if (Input(0).template IsType<float16>()) RunWithType<float16>();
#endif
    else LOG(FATAL) << "Unsupported input types.";
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
    kernel::TransposeGrad<T, Context>(Input(-1).count(), order->count(),
                                      ORdata, OSdata, NSdata,
                                      dYdata, dXdata);
}

template <class Context>
void TransposeGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));
    order = ws()->GetTensor("/mnt/" + Anchor() + "/transpose/order");
    old_steps = ws()->GetTensor("/mnt/" + Anchor() + "/transpose/old_steps");
    new_steps = ws()->GetTensor("/mnt/" + Anchor() + "/transpose/new_steps");

    if (Input(0).template IsType<float>()) RunWithType<float>();
#ifdef WITH_CUDA_FP16
    else if (Input(0).template IsType<float16>()) RunWithType<float16>();
#endif
    else LOG(FATAL) << "Unsupported input types.";
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

}    // namespace dragon