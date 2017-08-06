#include "operators/arithmetic/gram_matrix_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context> template <typename T>
void GramMatrixOp<Context>::RunWithType() {
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    for (int i = 0; i < outer_dim; i++) {
        math::Gemm<T, Context>(CblasNoTrans, CblasTrans,
            dim, dim, inner_dim, 1.0, Xdata, Xdata, 0.0, Ydata);
        Xdata += x_offset;
        Ydata += y_offset;
    }
}

template <class Context>
void GramMatrixOp<Context>::RunOnDevice() {
    outer_dim = input(0).count(0, axis);
    dim = input(0).dim(axis);
    inner_dim = input(0).count(axis + 1);
    x_offset = dim * inner_dim, y_offset = dim * dim;
    output(0)->Reshape(vector<TIndex>({ outer_dim, dim, dim }));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else if (input(0).template IsType<float16>()) RunWithType<float16>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(GramMatrix);
#ifdef WITH_CUDA
DEPLOY_CUDA(GramMatrix);
#endif
OPERATOR_SCHEMA(GramMatrix).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void GramMatrixGradientOp<Context>::RunWithType() {
    auto* dYdata = input(-1).template data<T, Context>();
    auto* Xdata = input(0).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    for (int i = 0; i < outer_dim; i++) {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans,
            dim, inner_dim, dim, 2.0, dYdata, Xdata, 0.0, dXdata);
        dYdata += y_offset;
        dXdata += x_offset;
    }
}

template <class Context>
void GramMatrixGradientOp<Context>::RunOnDevice() {
    outer_dim = input(0).count(0, axis);
    dim = input(0).dim(axis);
    inner_dim = input(0).count(axis + 1);
    x_offset = dim * inner_dim, y_offset = dim * dim;
    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else if (input(0).template IsType<float16>()) RunWithType<float16>();
    else LOG(FATAL) << "unsupported input types.";
}

template <class Context>
void GramMatrixGradientOp<Context>::ShareBeforeRun() {
    Tensor* dX = ws()->GetBuffer();
    if (dX != nullptr) output(0)->Replace(*dX);
}

template <class Context>
void GramMatrixGradientOp<Context>::ClearAfterRun() {
    Tensor* dY = &input(-1);
    ws()->ReleaseBuffer(dY);
}

DEPLOY_CPU(GramMatrixGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(GramMatrixGradient);
#endif
OPERATOR_SCHEMA(GramMatrixGradient).NumInputs(2).NumOutputs(1);

class GetGramMatrixGradient final : public GradientMakerBase {
public:
    GRADIENT_MAKER_CTOR(GetGramMatrixGradient);
    vector<OperatorDef> MakeDefs() override{
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(GramMatrix, GetGramMatrixGradient);

}    // namespace dragon