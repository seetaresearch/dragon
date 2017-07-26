#include "operators/arithmetic/matmul_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context> template <typename T>
void MatmulOp<Context>::RunWithType() {
    n = input(0).count() / M / K1;
    x1_offset = M * K1, x2_offset = K2 * N, y_offset = M * N;
    auto* X1data = input(0).template data<T, Context>();
    auto* X2data = input(1).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    for (int i = 0; i < n; i++) {
        math::Gemm<T, Context>(transA ? CblasTrans : CblasNoTrans,
                               transB ? CblasTrans : CblasNoTrans,
                               M, N, K1, 1.0, X1data, X2data, 0.0, Ydata);
        X1data += x1_offset;
        X2data += x2_offset;
        Ydata += y_offset;
    }
}

template <class Context>
void MatmulOp<Context>::RunOnDevice() {
    CHECK(input(0).ndim() == input(1).ndim())
        << "both matrices must have the same number of dimensions.";
    CHECK_GE(input(0).ndim(), 2)
        << "Tensor(" << input(0).name() + ") must be a matrix"
        << "(or rank > 2, representing batches of matrices).";
    CHECK_GE(input(1).ndim(), 2)
        << "Tensor(" << input(1).name() + ") must be a matrix"
        << "(or rank > 2, representing batches of matrices).";
    TIndex m = input(0).dim(-2), k = input(0).dim(-1);
    M = transA ? k : m;
    K1 = transA ? m : k;
    K2 = transB ? input(1).dim(-1) : input(1).dim(-2);
    N = transB ? input(1).dim(-2) : input(1).dim(-1);
    CHECK_EQ(K1, K2) << "\nTensor(" << input(0).name() << "): "
                     << input(0).dim_string() << " can not mul with Tensor"
                     << "(" << input(1).name() << "): " << input(1).dim_string();
    CHECK_EQ(input(0).count() / M / K1, input(1).count() / K2 / N)
                     << "\nTensor(" << input(0).name() << "): "
                     << input(0).dim_string() << " can not mul with Tensor"
                     << "(" << input(1).name() << "): " << input(1).dim_string();
    vector<TIndex> dims = input(0).dims();
    dims[dims.size() - 2] = M;
    dims[dims.size() - 1] = N;
    output(0)->Reshape(dims);
    if (input(0).template IsType<float>()) RunWithType<float>();
    else if (input(0).template IsType<float16>()) RunWithType<float16>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(Matmul);
#ifdef WITH_CUDA
DEPLOY_CUDA(Matmul);
#endif
OPERATOR_SCHEMA(Matmul).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void MatmulGradientOp<Context>::RunWithType() {
    n = input(0).count() / M / K1;
    x1_offset = M * K1, x2_offset = K2 * N, y_offset = M * N;
    auto* X1data = input(0).template data<T, Context>();
    auto* X2data = input(1).template data<T, Context>();
    auto* dYdata = input(2).template data<T, Context>();
    auto* dX1data = output(0)->template mutable_data<T, Context>();
    auto* dX2data = output(1)->template mutable_data<T, Context>();
    for (int i = 0; i < n; i++) {
        math::Gemm<T, Context>(CblasNoTrans, transB ? CblasNoTrans : CblasTrans,
                               M, K1, N, 1.0, dYdata, X2data, 0.0, dX1data);
        math::Gemm<T, Context>(transA ? CblasNoTrans : CblasTrans, CblasNoTrans,
                               K1, N, M, 1.0, X1data, dYdata, 0.0, dX2data);
        X1data += x1_offset;
        X2data += x2_offset;
        dX1data += x1_offset;
        dX2data += x2_offset;
        dYdata += y_offset;
    }
}

template <class Context>
void MatmulGradientOp<Context>::RunOnDevice(){
    CHECK(input(0).ndim() == input(1).ndim())
        << "both matrices must have the same number of dimensions.";
    CHECK_GE(input(0).ndim(), 2)
        << "Tensor(" << input(0).name() + ") must be a matrix"
        << "(or rank > 2, representing batches of matrices).";
    CHECK_GE(input(1).ndim(), 2)
        << "Tensor(" << input(1).name() + ") must be a matrix"
        << "(or rank > 2, representing batches of matrices).";
    TIndex m = input(0).dim(-2), k = input(0).dim(-1);
    M = transA ? k : m;
    K1 = transA ? m : k;
    K2 = transB ? input(1).dim(-1) : input(1).dim(-2);
    N = transB ? input(1).dim(-2) : input(1).dim(-1);
    CHECK_EQ(K1, K2) << "\nTensor(" << input(0).name() << "): "
        << input(0).dim_string() << " can not mul with Tensor"
        << "(" << input(1).name() << "): " << input(1).dim_string();
    CHECK_EQ(input(0).count() / M / K1, input(1).count() / K2 / N)
        << "\nTensor(" << input(0).name() << "): "
        << input(0).dim_string() << " can not mul with Tensor"
        << "(" << input(1).name() << "): " << input(1).dim_string();
    output(0)->ReshapeLike(input(0));
    output(1)->ReshapeLike(input(1));
    if (input(0).template IsType<float>()) RunWithType<float>();
    else if (input(0).template IsType<float16>()) RunWithType<float16>();
    else LOG(FATAL) << "unsupported input types.";
}

template <class Context>
void MatmulGradientOp<Context>::ShareBeforeRun() {
    for (int i = 0; i < OutputSize(); i++) {
        if (output(i)->name() != "ignore") {
            Tensor* dX = ws()->GetBuffer();
            if (dX != nullptr) output(i)->Replace(*dX);
            break;
        }
    }
}

template <class Context>
void MatmulGradientOp<Context>::ClearAfterRun() {
    Tensor* dY = &input(-1);
    ws()->ReleaseBuffer(dY);
}

DEPLOY_CPU(MatmulGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(MatmulGradient);
#endif
OPERATOR_SCHEMA(MatmulGradient).NumInputs(3).NumOutputs(2);

class GetMatmulGradient final : public GradientMakerBase {
public:
    GRADIENT_MAKER_CTOR(GetMatmulGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0), GI(1)});
    }
};
REGISTER_GRADIENT(Matmul, GetMatmulGradient);

}   // namespace dragon