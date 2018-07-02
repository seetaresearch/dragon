#include "operators/arithmetic/dot_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h" 

namespace dragon {

template <class Context> template <typename T>
void DotOp<Context>::DotRunWithType() {
    auto* X1data = Input(0).template data<T, Context>();
    auto* X2data = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, CPUContext>();
    Ydata[0] = math::Dot<T, Context>(
        Input(0).count(), X1data, X2data, &ctx());
}

template <class Context> template <typename T>
void DotOp<Context>::GemmRunWithType() {
    auto* X1data = Input(0).template data<T, Context>();
    auto* X2data = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    math::Gemm<T, Context>(
        TransA ? CblasTrans : CblasNoTrans,
            TransB ? CblasTrans : CblasNoTrans,
                M, N1, K1,
                    1.0, X1data, X2data,
                        0.0, Ydata, &ctx());
}

template <class Context> template <typename T>
void DotOp<Context>::GemvRunWithType() {
    auto* X1data = Input(0).template data<T, Context>();
    auto* X2data = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    math::Gemv<T, Context>(
        TransA ? CblasTrans : CblasNoTrans, M, N1,
            1.0, X1data, X2data,
                0.0, Ydata, &ctx());
}

template <class Context>
void DotOp<Context>::RunOnDevice() {
    if (Input(0).ndim() == 1 && Input(1).ndim() == 1) {
        CHECK_EQ(Input(0).dim(0), Input(1).dim(0))
            << "\nTensor(" << Input(0).name() << "): "
            << Input(0).DimString() << " can not Dot with Tensor"
            << "(" << Input(1).name() << "): " << Input(1).DimString();
        Output(0)->Reshape({ 1 });
        if (XIsType(Input(0), float)) DotRunWithType<float>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
    } 
    else if (Input(0).ndim() >= 2 && Input(1).ndim() == 2) {
        TIndex m = Input(0).count() / Input(0).dim(-1), k = Input(0).dim(-1);
        M =  TransA ? k : m;
        K1 = TransA ? m : k;
        K2 = TransB ? Input(1).dim(1) : Input(1).dim(0);
        N1 =  TransB ? Input(1).dim(0) : Input(1).dim(1);
        CHECK_EQ(K1, K2) << "\nTensor(" << Input(0).name() << "): "
            << Input(0).DimString() << " can not Dot with Tensor"
            << "(" << Input(1).name() << "): " << Input(1).DimString();
        vector<TIndex> dims = Input(0).dims();
        dims[dims.size() - 1] = N1;
        Output(0)->Reshape(dims);
        if (XIsType(Input(0), float)) GemmRunWithType<float>();
        else if (XIsType(Input(0), float16)) GemmRunWithType<float16>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
    } 
    else if (Input(0).ndim() >= 2 && Input(1).ndim() == 1) {
        TIndex m = Input(0).count() / Input(0).dim(-1), k = Input(0).dim(-1);
        M = TransA ? k : m;
        N1 = TransA ? m : k;
        N2 = Input(1).dim(0);
        CHECK_EQ(N1, N2) << "\nTensor(" << Input(0).name() << "): "
            << Input(0).DimString() << " can not Dot with Tensor"
            << "(" << Input(1).name() << "): " << Input(1).DimString();
        vector<TIndex> dims = Input(0).dims();
        dims.pop_back();
        Output(0)->Reshape(dims);
        if (XIsType(Input(0), float)) GemvRunWithType<float>();
        else if (XIsType(Input(0), float16)) GemvRunWithType<float16>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
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
    auto* X1data = Input(0).template data<T, Context>();
    auto* X2data = Input(1).template data<T, Context>();
    auto* dYdata = Input(2).template data<T, CPUContext>();
    auto* dX1data = Output(0)->template mutable_data<T, Context>();
    auto* dX2data = Output(1)->template mutable_data<T, Context>();
    this->ctx().template Copy<T, Context, Context>(
        Output(0)->count(), dX1data, X2data);
    this->ctx().template Copy<T, Context, Context>(
        Output(1)->count(), dX2data, X1data);
    math::MulScalar<T, Context>(Output(0)->count(), dYdata[0], dX1data);
    math::MulScalar<T, Context>(Output(1)->count(), dYdata[0], dX2data);
}

template <class Context> template <typename T>
void DotGradientOp<Context>::GemmRunWithType() {
    auto* X1data = Input(0).template data<T, Context>();
    auto* X2data = Input(1).template data<T, Context>();
    auto* dYdata = Input(2).template data<T, Context>();
    auto* dX1data = Output(0)->template mutable_data<T, Context>();
    auto* dX2data = Output(1)->template mutable_data<T, Context>();
    math::Gemm<T, Context>(
        CblasNoTrans,
            TransB ? CblasNoTrans : CblasTrans,
                M, K1, N1,
                    1.0, dYdata, X2data,
                        0.0, dX1data, &ctx());
    math::Gemm<T, Context>(
        TransA ? CblasNoTrans : CblasTrans,
            CblasNoTrans,
                K1, N1, M,
                    1.0, X1data, dYdata,
                        0.0, dX2data, &ctx());
}

template <class Context> template <typename T>
void DotGradientOp<Context>::GemvRunWithType() {
    auto* X1data = Input(0).template data<T, Context>();
    auto* X2data = Input(1).template data<T, Context>();
    auto* dYdata = Input(2).template data<T, Context>();
    auto* dX1data = Output(0)->template mutable_data<T, Context>();
    auto* dX2data = Output(1)->template mutable_data<T, Context>();
    math::Gemm<T, Context>(
        CblasNoTrans, CblasNoTrans,
            M, N1, 1,
                1.0, dYdata, X2data,
                    0.0, dX1data, &ctx());
    math::Gemv<T, Context>(
        TransA ? CblasNoTrans : CblasTrans, M, N1,
            1.0, X1data, dYdata,
                0.0, dX2data, &ctx());
}

template <class Context>
void DotGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));
    Output(1)->ReshapeLike(Input(1));

    if (Input(0).ndim() == 1 && Input(1).ndim() == 1) {
        CHECK_EQ(Input(0).dim(0), Input(1).dim(0))
            << "\nTensor(" << Input(0).name() << "): "
            << Input(0).DimString() << " can not Dot with Tensor"
            << "(" << Input(1).name() << "): " << Input(1).DimString();
        if (XIsType(Input(0), float)) DotRunWithType<float>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
    } 
    else if (Input(0).ndim() >= 2 && Input(1).ndim() == 2) {
        TIndex m = Input(0).count() / Input(0).dim(-1), k = Input(0).dim(-1);
        M =  TransA ? k : m;
        K1 = TransA ? m : k;
        K2 = TransB ? Input(1).dim(1) : Input(1).dim(0);
        N1 =  TransB ? Input(1).dim(0) : Input(1).dim(1);
        CHECK_EQ(K1, K2) << "\nTensor(" << Input(0).name() << "): "
            << Input(0).DimString() << " can not Dot with Tensor"
            << "(" << Input(1).name() << "): " << Input(1).DimString();
        if (XIsType(Input(0), float)) GemmRunWithType<float>();
        else if (XIsType(Input(0), float16)) GemmRunWithType<float16>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
    } 
    else if (Input(0).ndim() >= 2 && Input(1).ndim() == 1) {
        TIndex m = Input(0).count() / Input(0).dim(-1), k = Input(0).dim(-1);
        M = TransA ? k : m;
        N1 = TransA ? m : k;
        N2 = Input(1).dim(0);
        CHECK_EQ(N1, N2) << "\nTensor(" << Input(0).name() << "): "
            << Input(0).DimString() << " can not Dot with Tensor"
            << "(" << Input(1).name() << "): " << Input(1).DimString();
        if (XIsType(Input(0), float)) GemvRunWithType<float>();
        else if (XIsType(Input(0), float16)) GemvRunWithType<float16>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
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
OPERATOR_SCHEMA(DotGradient).NumInputs(3).NumOutputs(2);

class GetDotGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetDotGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0), GI(1)});
    }
};
REGISTER_GRADIENT(Dot, GetDotGradient);

}    // namespace dragon