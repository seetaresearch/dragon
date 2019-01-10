#include "utils/math_functions.h"
#include "operators/arithmetic/matmul_op.h"

namespace dragon {

template <class Context> template <typename T>
void MatmulOp<Context>::RunWithType() {
    auto* Adata = Input(0).template data<T, Context>();
    auto* Bdata = Input(1).template data<T, Context>();
    auto* Cdata = Output(0)->template mutable_data<T, Context>();

    for (int i = 0; i < batch_size; ++i) {
        math::Gemm(
            transA ? CblasTrans : CblasNoTrans,
            transB ? CblasTrans : CblasNoTrans,
            M, N, K1,
            1.f, Adata + i * A_stride, Bdata + i * B_stride,
            0.f, Cdata + i * C_stride, ctx());
    }
}

template <class Context>
void MatmulOp<Context>::RunOnDevice() {
    CHECK_GE(Input(0).ndim(), 2)
        << "\nTensor(" << Input(0).name() + ") must be a matrix"
        << "(or rank > 2, representing batches of matrices).";
    CHECK_GE(Input(1).ndim(), 2)
        << "\nTensor(" << Input(1).name() + ") must be a matrix"
        << "(or rank > 2, representing batches of matrices).";

    M1 = Input(0).dim(-2), N1 = Input(0).dim(-1);
    M2 = Input(1).dim(-2), N2 = Input(1).dim(-1);

    M = transA ? N1 : M1, N = transB ? M2 : N2;
    K1 = transA ? M1 : N1, K2 = transB ? N2 : M2;
    A_stride = M1 * N1, B_stride = M2 * N2, C_stride = M * N;
    batch_size = Input(0).count() / A_stride;

    CHECK_EQ(K1, K2) << "\nTensor(" << Input(0).name() << "): "
        << Input(0).DimString() << " can not mul with Tensor"
        << "(" << Input(1).name() << "): " << Input(1).DimString();

    CHECK_EQ(batch_size, Input(1).count() / B_stride)
        << "\nTensor(" << Input(0).name() << "): "
        << Input(0).DimString() << " can not mul with Tensor"
        << "(" << Input(1).name() << "): " << Input(1).DimString();

    vector<int64_t> dims = Input(0).dims();
    dims[dims.size() - 2] = M; dims[dims.size() - 1] = N;
    Output(0)->Reshape(dims);

    if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0), {
        "float16", "float32", "float64" });
}

DEPLOY_CPU(Matmul);
#ifdef WITH_CUDA
DEPLOY_CUDA(Matmul);
#endif
OPERATOR_SCHEMA(Matmul).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void MatmulGradientOp<Context>::RunWithType() {
    auto* Adata = Input(0).template data<T, Context>();
    auto* Bdata = Input(1).template data<T, Context>();
    auto* dCdata = Input(-1).template data<T, Context>();

    T* dAdata = nullptr, *dBdata = nullptr;

    if (Output(0)->name() != "ignore") {
        dAdata = Output(0)->template mutable_data<T, Context>();
    } if (Output(1)->name() != "ignore") {
        dBdata = Output(1)->template mutable_data<T, Context>();
    }

    for (int i = 0; i < batch_size; ++i) {
        if (Output(0)->name() != "ignore") {
            if (transA) {
                math::Gemm(
                    transB ? CblasTrans : CblasNoTrans,
                    CblasTrans,
                    K1, M, N,
                    1.f, Bdata + i * B_stride, dCdata + i * C_stride,
                    0.f, dAdata + i * A_stride, ctx());
            } else {
                math::Gemm(
                    CblasNoTrans,
                    transB ? CblasNoTrans : CblasTrans,
                    M, K1, N,
                    1.f, dCdata + i * C_stride, Bdata + i * B_stride,
                    0.f, dAdata + i * A_stride, ctx());
            }
        }
        if (Output(1)->name() != "ignore") {
            if (transB) {
                math::Gemm(
                    CblasTrans,
                    transA ? CblasTrans : CblasNoTrans,
                    N, K1, M,
                    1.f, dCdata + i * C_stride, Adata + i * A_stride,
                    0.f, dBdata + i * B_stride, ctx());
            } else {
                math::Gemm(
                    transA ? CblasNoTrans : CblasTrans,
                    CblasNoTrans,
                    K1, N, M,
                    1.f, Adata + i * A_stride, dCdata + i * C_stride,
                    0.f, dBdata + i * B_stride, ctx());
            }
        }
    }
}

template <class Context>
void MatmulGradientOp<Context>::RunOnDevice() {
    CHECK_GE(Input(0).ndim(), 2)
        << "\nTensor(" << Input(0).name() + ") must be a matrix"
        << "(or rank > 2, representing batches of matrices).";
    CHECK_GE(Input(1).ndim(), 2)
        << "\nTensor(" << Input(1).name() + ") must be a matrix"
        << "(or rank > 2, representing batches of matrices).";

    M1 = Input(0).dim(-2), N1 = Input(0).dim(-1);
    M2 = Input(1).dim(-2), N2 = Input(1).dim(-1);

    M = transA ? N1 : M1, N = transB ? M2 : N2;
    K1 = transA ? M1 : N1, K2 = transB ? N2 : M2;
    A_stride = M1 * N1, B_stride = M2 * N2, C_stride = M * N;
    batch_size = Input(0).count() / A_stride;

    CHECK_EQ(K1, K2) << "\nTensor(" << Input(0).name() << "): "
        << Input(0).DimString() << " can not mul with Tensor"
        << "(" << Input(1).name() << "): " << Input(1).DimString();

    CHECK_EQ(batch_size, Input(1).count() / B_stride)
        << "\nTensor(" << Input(0).name() << "): "
        << Input(0).DimString() << " can not mul with Tensor"
        << "(" << Input(1).name() << "): " << Input(1).DimString();

    Output(0)->ReshapeLike(Input(0));
    Output(1)->ReshapeLike(Input(1));

    if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0), {
        "float16", "float32", "float64" });
}

DEPLOY_CPU(MatmulGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(MatmulGradient);
#endif

OPERATOR_SCHEMA(MatmulGradient)
    .NumInputs(3).NumOutputs(2);

REGISTER_GRADIENT(Matmul, SimpleGradientMaker);

}  // namespace dragon