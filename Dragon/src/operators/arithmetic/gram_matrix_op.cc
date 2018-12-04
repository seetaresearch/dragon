#include "utils/math_functions.h"
#include "operators/arithmetic/gram_matrix_op.h"

namespace dragon {

template <class Context> template <typename T>
void GramMatrixOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    for (int i = 0; i < outer_dim; i++) {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasTrans,
                dim, dim, inner_dim,
                    1.f, Xdata, Xdata,
                        0.f, Ydata, ctx());
        Xdata += x_offset;
        Ydata += y_offset;
    }
}

template <class Context>
void GramMatrixOp<Context>::RunOnDevice() {
    outer_dim = Input(0).count(0, axis);
    dim = Input(0).dim(axis);
    inner_dim = Input(0).count(axis + 1);
    x_offset = dim * inner_dim, y_offset = dim * dim;
    Output(0)->Reshape(vector<TIndex>({ outer_dim, dim, dim }));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(GramMatrix);
#ifdef WITH_CUDA
DEPLOY_CUDA(GramMatrix);
#endif
OPERATOR_SCHEMA(GramMatrix).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void GramMatrixGradientOp<Context>::RunWithType() {
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* Xdata = Input(0).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    for (int i = 0; i < outer_dim; i++) {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                dim, inner_dim, dim,
                    2.f, dYdata, Xdata,
                        0.f, dXdata, ctx());
        dYdata += y_offset;
        dXdata += x_offset;
    }
}

template <class Context>
void GramMatrixGradientOp<Context>::RunOnDevice() {
    outer_dim = Input(0).count(0, axis);
    dim = Input(0).dim(axis);
    inner_dim = Input(0).count(axis + 1);
    x_offset = dim * inner_dim, y_offset = dim * dim;
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
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

}  // namespace dragon