#include "operators/common/transpose_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void TransposeOp<Context>::RunWithType() {
    auto* Xdata = input(0).template data<T, Context>();
    auto* OSdata = old_steps->template data<int, Context>();
    auto* NSdata = new_steps->template data<int, Context>();
    auto* ORdata = order->template data <int, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    kernel::Transpose<T, Context>(output(0)->count(), int(perm.size()),
                                  ORdata, OSdata, NSdata, 
                                  Xdata, Ydata);
}

template <class Context>
void TransposeOp<Context>::RunOnDevice() {
    CHECK_EQ(input(0).ndim(), perm.size())
        << "\nprovide " << perm.size() << "perm to permute"
        << "\nbut Tensor(" << input(0).name() << ")' dims is "
        << input(0).dim_string();
    vector<TIndex> output_dims;
    order = ws()->CreateTensor("_t_" + anchor() + "_order");
    old_steps = ws()->CreateTensor("_t_" + anchor() + "_old_steps");
    new_steps = ws()->CreateTensor("_t_" + anchor() + "_new_steps");
    order->Reshape(vector<TIndex>(1, perm.size()));
    old_steps->Reshape(vector<TIndex>(1, perm.size()));
    new_steps->Reshape(vector<TIndex>(1, perm.size()));
    auto* OSdata = old_steps->template mutable_data<int, CPUContext>();
    auto* NSdata = new_steps->template mutable_data<int, CPUContext>();
    auto* ORdata = order->template mutable_data<int, CPUContext>();
    for (int i = 0; i < perm.size(); i++) {
        if (i == perm.size() - 1) OSdata[i] = 1;
        else OSdata[i] = input(0).count(i + 1);
        ORdata[i] = perm[i];
        output_dims.push_back(input(0).dim(perm[i]));
    }
    output(0)->Reshape(output_dims);
    for (int i = 0; i < perm.size(); i++) {
        if (i == perm.size() - 1) NSdata[i] = 1;
        else NSdata[i] = output(0)->count(i + 1);
    }

    if (input(0).template IsType<float>()) RunWithType<float>();
    else if (input(0).template IsType<float16>()) RunWithType<float16>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(Transpose);
#ifdef WITH_CUDA
DEPLOY_CUDA(Transpose);
#endif
OPERATOR_SCHEMA(Transpose).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void TransposeGradientOp<Context>::RunWithType() {
    auto* dYdata = input(-1).template data<T, Context>();
    auto* OSdata = old_steps->template data<int, Context>();
    auto* NSdata = new_steps->template data<int, Context>();
    auto* ORdata = order->template data <int, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    kernel::TransposeGrad<T, Context>(input(-1).count(), order->count(),
                                      ORdata, OSdata, NSdata,
                                      dYdata, dXdata);
}

template <class Context>
void TransposeGradientOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));
    order = ws()->GetTensor("_t_" + anchor() + "_order");
    old_steps = ws()->GetTensor("_t_" + anchor() + "_old_steps");
    new_steps = ws()->GetTensor("_t_" + anchor() + "_new_steps");

    if (input(0).template IsType<float>()) RunWithType<float>();
    else if (input(0).template IsType<float16>()) RunWithType<float16>();
    else LOG(FATAL) << "unsupported input types.";
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