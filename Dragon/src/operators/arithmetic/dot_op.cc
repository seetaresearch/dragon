#include "utils/math_functions.h"
#include "operators/arithmetic/dot_op.h"

namespace dragon {

template <class Context> template <typename T>
void DotOp<Context>::DotRunWithType() {
    CHECK_EQ(Input(0).dim(0), Input(1).dim(0))
        << "\nTensor(" << Input(0).name() << "): "
        << Input(0).DimString() << " can not Dot with Tensor"
        << "(" << Input(1).name() << "): " << Input(1).DimString();

    Output(0)->Reshape({});

    auto* X1data = Input(0).template data<T, Context>();
    auto* X2data = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    T YHost;
    math::Dot(Input(0).count(), X1data, X2data, &YHost, ctx());
    ctx()->template Copy<T, Context, CPUContext>(1, Ydata, &YHost);
}

template <class Context> template <typename T>
void DotOp<Context>::GemmRunWithType() {
    M1 = Input(0).count() / Input(0).dim(-1), N1 = Input(0).dim(-1),
    M2 = Input(1).dim(0), N2 = Input(1).dim(1);

    M = transA ? N1 : M1, N = transB ? M2 : N2;
    K1 = transA ? M1 : N1, K2 = transB ? N2 : M2;

    CHECK_EQ(K1, K2) << "\nTensor(" << Input(0).name() << "): "
        << Input(0).DimString() << " can not Dot with Tensor"
        << "(" << Input(1).name() << "): " << Input(1).DimString();

    auto dims = Input(0).dims(); dims.back() = N;
    Output(0)->Reshape(transA ? vector<int64_t>({ M, N }) : dims);

    auto* X1data = Input(0).template data<T, Context>();
    auto* X2data = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    math::Gemm(
        transA ? CblasTrans : CblasNoTrans,
            transB ? CblasTrans : CblasNoTrans,
                M, N, K1,
                    1.f, X1data, X2data,
                        0.f, Ydata, ctx());
}

template <class Context> template <typename T>
void DotOp<Context>::GemvRunWithType() {
    M1 = Input(0).count() / Input(0).dim(-1), N1 = Input(0).dim(-1);
    K1 = transA ? M1 : N1, K2 = Input(1).dim(0);

    CHECK_EQ(K1, K2) << "\nTensor(" << Input(0).name() << "): "
        << Input(0).DimString() << " can not Dot with Tensor"
        << "(" << Input(1).name() << "): " << Input(1).DimString();

    auto dims = Input(0).dims(); dims.pop_back();
    Output(0)->Reshape(transA ? vector<int64_t>({ N1 }) : dims);

    auto* X1data = Input(0).template data<T, Context>();
    auto* X2data = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    math::Gemv(
        transA ? CblasTrans : CblasNoTrans, M1, N1,
            1.f, X1data, X2data, 0.f, Ydata, ctx());
}

template <class Context>
void DotOp<Context>::RunOnDevice() {
    if (Input(0).ndim() == 1 && Input(1).ndim() == 1) {
        if (XIsType(Input(0), float16)) DotRunWithType<float16>();
        else if (XIsType(Input(0), float)) DotRunWithType<float>();
        else if (XIsType(Input(0), double)) DotRunWithType<double>();
        else LOG(FATAL) << DTypeHelper(Input(0), {
            "float16", "float32", "float64" });
    }
    else if (Input(0).ndim() >= 2 && Input(1).ndim() == 2) {
        if (XIsType(Input(0), float16)) GemmRunWithType<float16>();
        else if (XIsType(Input(0), float)) GemmRunWithType<float>();
        else if (XIsType(Input(0), double)) GemmRunWithType<double>();
        else LOG(FATAL) << DTypeHelper(Input(0), {
            "float16", "float32", "float64" });
    } 
    else if (Input(0).ndim() >= 2 && Input(1).ndim() == 1) {
        if (XIsType(Input(0), float16)) GemvRunWithType<float16>();
        else if (XIsType(Input(0), float)) GemvRunWithType<float>();
        else if (XIsType(Input(0), double)) GemvRunWithType<double>();
        else LOG(FATAL) << DTypeHelper(Input(0), {
            "float16", "float32", "float64" });
    } 
    else {
        LOG(FATAL) << "\nTensor(" << Input(0).name() << "): "
            << Input(0).DimString() << " can not Dot with Tensor"
            << "(" << Input(1).name() << "): " << Input(1).DimString();
    }
}

DEPLOY_CPU(Dot);
#ifdef WITH_CUDA
DEPLOY_CUDA(Dot);
#endif
OPERATOR_SCHEMA(Dot).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void DotGradientOp<Context>::DotRunWithType() {
    CHECK_EQ(Input(0).count(), Input(1).count())
        << "\nTensor(" << Input(0).name() << "): "
        << Input(0).DimString() << " can not Dot with Tensor"
        << "(" << Input(1).name() << "): " << Input(1).DimString();

    auto* Adata = Input(0).template data<T, Context>();
    auto* Bdata = Input(1).template data<T, Context>();
    auto* dYdata = Input(-1).template data<T, CPUContext>();

    if (Output(0)->name() != "ignore") {
        auto* dAdata = Output(0)->template mutable_data<T, Context>();
        math::Scale(Output(0)->count(), cast::to<float>(
            dYdata[0]), Bdata, dAdata, ctx());
    }

    if (Output(1)->name() != "ignore") {
        auto* dBdata = Output(1)->template mutable_data<T, Context>();
        math::Scale(Output(0)->count(), cast::to<float>(
            dYdata[0]), Adata, dBdata, ctx());
    }
}

template <class Context> template <typename T>
void DotGradientOp<Context>::GemmRunWithType() {
    M1 = Input(0).count() / Input(0).dim(-1), N1 = Input(0).dim(-1),
    M2 = Input(1).dim(0), N2 = Input(1).dim(1);

    M = transA ? N1 : M1, N = transB ? M2 : N2;
    K1 = transA ? M1 : N1, K2 = transB ? N2 : M2;

    CHECK_EQ(K1, K2) << "\nTensor(" << Input(0).name() << "): "
        << Input(0).DimString() << " can not Dot with Tensor"
        << "(" << Input(1).name() << "): " << Input(1).DimString();

    auto* X1data = Input(0).template data<T, Context>();
    auto* X2data = Input(1).template data<T, Context>();
    auto* dYdata = Input(2).template data<T, Context>();

    if (Output(0)->name() != "ignore") {
        auto* dX1data = Output(0)->template mutable_data<T, Context>();
        if (transA) {
            math::Gemm(
                transB ? CblasTrans : CblasNoTrans, CblasTrans,
                    K1, M, N,
                        1.f, X2data, dYdata,
                            0.f, dX1data, ctx());
        } else {
            math::Gemm(
                CblasNoTrans, transB ? CblasNoTrans : CblasTrans,
                    M, K1, N,
                        1.f, dYdata, X2data,
                            0.f, dX1data, ctx());
        }
    }

    if (Output(1)->name() != "ignore") {
        auto* dX2data = Output(1)->template mutable_data<T, Context>();
        if (transB) {
           math::Gemm(
                CblasTrans, transA ? CblasTrans : CblasNoTrans,
                    N, K1, M,
                        1.f, dYdata, X1data,
                            0.f, dX2data, ctx());
        } else {
            math::Gemm(
                transA ? CblasNoTrans : CblasTrans, CblasNoTrans,
                    K1, N, M,
                        1.f, X1data, dYdata,
                            0.f, dX2data, ctx());
        }
    }
}

template <class Context> template <typename T>
void DotGradientOp<Context>::GemvRunWithType() {
    M1 = Input(0).count() / Input(0).dim(-1), N1 = Input(0).dim(-1);
    K1 = transA ? M1 : N1, K2 = Input(1).dim(0);
    M = transA ? N1 : M1, N = K2;  // Keep M, Remove N

    CHECK_EQ(K1, K2) << "\nTensor(" << Input(0).name() << "): "
        << Input(0).DimString() << " can not Dot with Tensor"
        << "(" << Input(1).name() << "): " << Input(1).DimString();

    auto* X1data = Input(0).template data<T, Context>();
    auto* X2data = Input(1).template data<T, Context>();
    auto* dYdata = Input(2).template data<T, Context>();
    auto* dX1data = Output(0)->template mutable_data<T, Context>();
    auto* dX2data = Output(1)->template mutable_data<T, Context>();

    math::Gemm(
        CblasNoTrans, CblasNoTrans,
            M, N, 1,
                1.f, dYdata, X2data,
                    0.f, dX1data, ctx());

    math::Gemv(
        transA ? CblasNoTrans : CblasTrans,
            M1, N1,
                1.f, X1data, dYdata,
                    0.f, dX2data, ctx());
}

template <class Context>
void DotGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));
    Output(1)->ReshapeLike(Input(1));

    if (Input(0).ndim() == 1 && Input(1).ndim() == 1) {
        if (XIsType(Input(0), float16)) DotRunWithType<float16>();
        else if (XIsType(Input(0), float)) DotRunWithType<float>();
        else if (XIsType(Input(0), double)) DotRunWithType<double>();
        else LOG(FATAL) << DTypeHelper(Input(0), {
            "float16", "float32", "float64" });
    }
    else if (Input(0).ndim() >= 2 && Input(1).ndim() == 2) {
        if (XIsType(Input(0), float16)) GemmRunWithType<float16>();
        else if (XIsType(Input(0), float)) GemmRunWithType<float>();
        else if (XIsType(Input(0), double)) GemmRunWithType<double>();
        else LOG(FATAL) << DTypeHelper(Input(0), {
            "float16", "float32", "float64" });
    } 
    else if (Input(0).ndim() >= 2 && Input(1).ndim() == 1) {
        M = Input(0).count() / Input(0).dim(-1), K1 = Input(0).dim(-1);
        N = transA ? M : K1;
        CHECK_EQ(N, Input(1).dim(0)) << "\nTensor(" << Input(0).name() << "): "
            << Input(0).DimString() << " can not Dot with Tensor"
            << "(" << Input(1).name() << "): " << Input(1).DimString();
        if (XIsType(Input(0), float16)) GemvRunWithType<float16>();
        else if (XIsType(Input(0), float)) GemvRunWithType<float>();
        else if (XIsType(Input(0), double)) GemvRunWithType<double>();
        else LOG(FATAL) << DTypeHelper(Input(0), {
            "float16", "float32", "float64" });
    } 
    else {
        LOG(FATAL) << "\nTensor(" << Input(0).name() << "): "
            << Input(0).DimString() << " can not Dot with Tensor"
            << "(" << Input(1).name() << "): " << Input(1).DimString();
    }
}

DEPLOY_CPU(DotGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(DotGradient);
#endif

OPERATOR_SCHEMA(DotGradient)
    .NumInputs(3).NumOutputs(2);

REGISTER_GRADIENT(Dot, SimpleGradientMaker);

}  // namespace dragon