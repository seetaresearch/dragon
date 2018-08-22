#include "utils/math_functions.h"
#include "operators/arithmetic/matmul_op.h"

namespace dragon {

template <class Context> template <typename T>
void MatmulOp<Context>::RunWithType() {
    n = Input(0).count() / M / K1;
    x1_offset = M * K1, x2_offset = K2 * N, y_offset = M * N;
    auto* X1data = Input(0).template data<T, Context>();
    auto* X2data = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    for (int i = 0; i < n; i++) {
        math::Gemm<T, Context>(
            TransA ? CblasTrans : CblasNoTrans,
                TransB ? CblasTrans : CblasNoTrans,
                    M, N, K1,
                        1.0, X1data, X2data,
                            0.0, Ydata, ctx());
        X1data += x1_offset;
        X2data += x2_offset;
        Ydata += y_offset;
    }
}

template <class Context>
void MatmulOp<Context>::RunOnDevice() {
    CHECK(Input(0).ndim() == Input(1).ndim())
        << "Both matrices must have the same number of dimensions.";
    CHECK_GE(Input(0).ndim(), 2)
        << "Tensor(" << Input(0).name() + ") must be a matrix"
        << "(or rank > 2, representing batches of matrices).";
    CHECK_GE(Input(1).ndim(), 2)
        << "Tensor(" << Input(1).name() + ") must be a matrix"
        << "(or rank > 2, representing batches of matrices).";
    TIndex m = Input(0).dim(-2), k = Input(0).dim(-1);
    M = TransA ? k : m;
    K1 = TransA ? m : k;
    K2 = TransB ? Input(1).dim(-1) : Input(1).dim(-2);
    N = TransB ? Input(1).dim(-2) : Input(1).dim(-1);
    CHECK_EQ(K1, K2) << "\nTensor(" << Input(0).name() << "): "
                     << Input(0).DimString() << " can not mul with Tensor"
                     << "(" << Input(1).name() << "): " << Input(1).DimString();
    CHECK_EQ(Input(0).count() / M / K1, Input(1).count() / K2 / N)
                     << "\nTensor(" << Input(0).name() << "): "
                     << Input(0).DimString() << " can not mul with Tensor"
                     << "(" << Input(1).name() << "): " << Input(1).DimString();
    vector<TIndex> dims = Input(0).dims();
    dims[dims.size() - 2] = M;
    dims[dims.size() - 1] = N;
    Output(0)->Reshape(dims);

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(Matmul);
#ifdef WITH_CUDA
DEPLOY_CUDA(Matmul);
#endif
OPERATOR_SCHEMA(Matmul).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void MatmulGradientOp<Context>::RunWithType() {
    n = Input(0).count() / M / K1;
    x1_offset = M * K1, x2_offset = K2 * N, y_offset = M * N;
    auto* X1data = Input(0).template data<T, Context>();
    auto* X2data = Input(1).template data<T, Context>();
    auto* dYdata = Input(2).template data<T, Context>();
    auto* dX1data = Output(0)->template mutable_data<T, Context>();
    auto* dX2data = Output(1)->template mutable_data<T, Context>();
    for (int i = 0; i < n; i++) {
        math::Gemm<T, Context>(
            CblasNoTrans,
                TransB ? CblasNoTrans : CblasTrans,
                    M, K1, N,
                        1.0, dYdata, X2data,
                            0.0, dX1data, ctx());
        math::Gemm<T, Context>(
            TransA ? CblasNoTrans : CblasTrans,
                CblasNoTrans,
                    K1, N, M,
                        1.0, X1data, dYdata,
                            0.0, dX2data, ctx());
        X1data += x1_offset;
        X2data += x2_offset;
        dX1data += x1_offset;
        dX2data += x2_offset;
        dYdata += y_offset;
    }
}

template <class Context>
void MatmulGradientOp<Context>::RunOnDevice() {
    CHECK(Input(0).ndim() == Input(1).ndim())
        << "Both matrices must have the same number of dimensions.";
    CHECK_GE(Input(0).ndim(), 2)
        << "Tensor(" << Input(0).name() + ") must be a matrix"
        << "(or rank > 2, representing batches of matrices).";
    CHECK_GE(Input(1).ndim(), 2)
        << "Tensor(" << Input(1).name() + ") must be a matrix"
        << "(or rank > 2, representing batches of matrices).";
    TIndex m = Input(0).dim(-2), k = Input(0).dim(-1);
    M = TransA ? k : m;
    K1 = TransA ? m : k;
    K2 = TransB ? Input(1).dim(-1) : Input(1).dim(-2);
    N = TransB ? Input(1).dim(-2) : Input(1).dim(-1);
    CHECK_EQ(K1, K2) << "\nTensor(" << Input(0).name() << "): "
        << Input(0).DimString() << " can not mul with Tensor"
        << "(" << Input(1).name() << "): " << Input(1).DimString();
    CHECK_EQ(Input(0).count() / M / K1, Input(1).count() / K2 / N)
        << "\nTensor(" << Input(0).name() << "): "
        << Input(0).DimString() << " can not mul with Tensor"
        << "(" << Input(1).name() << "): " << Input(1).DimString();
    Output(0)->ReshapeLike(Input(0));
    Output(1)->ReshapeLike(Input(1));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
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