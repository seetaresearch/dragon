#include "operators/arithmetic/dot_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h" 

namespace dragon {

template <class Context> template <typename T>
void DotOp<Context>::DotRunWithType() {
    auto* X1data = input(0).template data<T, Context>();
    auto* X2data = input(1).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, CPUContext>();
    Ydata[0] = math::Dot<T, Context>(input(0).count(), X1data, X2data);
}

template <class Context> template <typename T>
void DotOp<Context>::GemmRunWithType() {
    auto* X1data = input(0).template data<T, Context>();
    auto* X2data = input(1).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    math::Gemm<T, Context>(transA ? CblasTrans : CblasNoTrans,
                           transB ? CblasTrans : CblasNoTrans,
                           M, N1, K1, 1.0, X1data, X2data, 0.0, Ydata);
}

template <class Context> template <typename T>
void DotOp<Context>::GemvRunWithType() {
    auto* X1data = input(0).template data<T, Context>();
    auto* X2data = input(1).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    math::Gemv<T, Context>(transA ? CblasTrans : CblasNoTrans,
                           M, N1, 1.0, X1data, X2data, 0.0, Ydata);
}

template <class Context>
void DotOp<Context>::RunOnDevice() {
    if (input(0).ndim() == 1 && input(1).ndim() == 1) {
        CHECK_EQ(input(0).dim(0), input(1).dim(0))
            << "\nTensor(" << input(0).name() << "): "
            << input(0).dim_string() << " can not Dot with Tensor"
            << "(" << input(1).name() << "): " << input(1).dim_string();
        output(0)->Reshape(vector<TIndex>(1, 1));
        if (input(0).template IsType<float>()) DotRunWithType<float>();
        else LOG(FATAL) << "Unsupported input types.";
    } 
    else if (input(0).ndim() >= 2 && input(1).ndim() == 2) {
        TIndex m = input(0).count() / input(0).dim(-1), k = input(0).dim(-1);
        M =  transA ? k : m;
        K1 = transA ? m : k;
        K2 = transB ? input(1).dim(1) : input(1).dim(0);
        N1 =  transB ? input(1).dim(0) : input(1).dim(1);
        CHECK_EQ(K1, K2) << "\nTensor(" << input(0).name() << "): "
            << input(0).dim_string() << " can not Dot with Tensor"
            << "(" << input(1).name() << "): " << input(1).dim_string();
        vector<TIndex> dims = input(0).dims();
        dims[dims.size() - 1] = N1;
        output(0)->Reshape(dims);
        if (input(0).template IsType<float>()) GemmRunWithType<float>();
#ifdef WITH_CUDA_FP16
        else if (input(0).template IsType<float16>()) GemmRunWithType<float16>();
#endif
        else LOG(FATAL) << "Unsupported input types.";
    } 
    else if (input(0).ndim() >= 2 && input(1).ndim() == 1) {
        TIndex m = input(0).count() / input(0).dim(-1), k = input(0).dim(-1);
        M = transA ? k : m;
        N1 = transA ? m : k;
        N2 = input(1).dim(0);
        CHECK_EQ(N1, N2) << "\nTensor(" << input(0).name() << "): "
            << input(0).dim_string() << " can not Dot with Tensor"
            << "(" << input(1).name() << "): " << input(1).dim_string();
        vector<TIndex> dims = input(0).dims();
        dims.pop_back();
        output(0)->Reshape(dims);
        if (input(0).template IsType<float>()) GemvRunWithType<float>();
#ifdef WITH_CUDA_FP16
        else if (input(0).template IsType<float16>()) GemvRunWithType<float16>();
#endif
        else LOG(FATAL) << "Unsupported input types.";
    } 
    else {
        LOG(FATAL) << "\nTensor(" << input(0).name() << "): "
            << input(0).dim_string() << " can not Dot with Tensor"
            << "(" << input(1).name() << "): " << input(1).dim_string();
    }
}

DEPLOY_CPU(Dot);
#ifdef WITH_CUDA
DEPLOY_CUDA(Dot);
#endif
OPERATOR_SCHEMA(Dot).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void DotGradientOp<Context>::DotRunWithType() {
    auto* X1data = input(0).template data<T, Context>();
    auto* X2data = input(1).template data<T, Context>();
    auto* dYdata = input(2).template data<T, CPUContext>();
    auto* dX1data = output(0)->template mutable_data<T, Context>();
    auto* dX2data = output(1)->template mutable_data<T, Context>();
    this->ctx().template Copy<T, Context, Context>(output(0)->count(), dX1data, X2data);
    this->ctx().template Copy<T, Context, Context>(output(1)->count(), dX2data, X1data);
    math::MulScalar<T, Context>(output(0)->count(), dYdata[0], dX1data);
    math::MulScalar<T, Context>(output(1)->count(), dYdata[0], dX2data);
}

template <class Context> template <typename T>
void DotGradientOp<Context>::GemmRunWithType() {
    auto* X1data = input(0).template data<T, Context>();
    auto* X2data = input(1).template data<T, Context>();
    auto* dYdata = input(2).template data<T, Context>();
    auto* dX1data = output(0)->template mutable_data<T, Context>();
    auto* dX2data = output(1)->template mutable_data<T, Context>();
    math::Gemm<T, Context>(CblasNoTrans, transB ? CblasNoTrans : CblasTrans,
                           M, K1, N1, 1.0, dYdata, X2data, 0.0, dX1data);
    math::Gemm<T, Context>(transA ? CblasNoTrans : CblasTrans, CblasNoTrans,
                           K1, N1, M, 1.0, X1data, dYdata, 0.0, dX2data);
}

template <class Context> template <typename T>
void DotGradientOp<Context>::GemvRunWithType() {
    auto* X1data = input(0).template data<T, Context>();
    auto* X2data = input(1).template data<T, Context>();
    auto* dYdata = input(2).template data<T, Context>();
    auto* dX1data = output(0)->template mutable_data<T, Context>();
    auto* dX2data = output(1)->template mutable_data<T, Context>();
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans,
                           M, N1, 1, 1.0, dYdata, X2data, 0.0, dX1data);
    math::Gemv<T, Context>(transA ? CblasNoTrans : CblasTrans,
                           M, N1, 1.0, X1data, dYdata, 0.0, dX2data);
}

template <class Context>
void DotGradientOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));
    output(1)->ReshapeLike(input(1));

    if (input(0).ndim() == 1 && input(1).ndim() == 1) {
        CHECK_EQ(input(0).dim(0), input(1).dim(0))
            << "\nTensor(" << input(0).name() << "): "
            << input(0).dim_string() << " can not Dot with Tensor"
            << "(" << input(1).name() << "): " << input(1).dim_string();
        if (input(0).template IsType<float>()) DotRunWithType<float>();
        else LOG(FATAL) << "Unsupported input types.";
    } 
    else if (input(0).ndim() >= 2 && input(1).ndim() == 2) {
        TIndex m = input(0).count() / input(0).dim(-1), k = input(0).dim(-1);
        M =  transA ? k : m;
        K1 = transA ? m : k;
        K2 = transB ? input(1).dim(1) : input(1).dim(0);
        N1 =  transB ? input(1).dim(0) : input(1).dim(1);
        CHECK_EQ(K1, K2) << "\nTensor(" << input(0).name() << "): "
            << input(0).dim_string() << " can not Dot with Tensor"
            << "(" << input(1).name() << "): " << input(1).dim_string();
        if (input(0).template IsType<float>()) GemmRunWithType<float>();
#ifdef WITH_CUDA_FP16
        else if (input(0).template IsType<float16>()) GemmRunWithType<float16>();
#endif
        else LOG(FATAL) << "Unsupported input types.";
    } 
    else if (input(0).ndim() >= 2 && input(1).ndim() == 1) {
        TIndex m = input(0).count() / input(0).dim(-1), k = input(0).dim(-1);
        M = transA ? k : m;
        N1 = transA ? m : k;
        N2 = input(1).dim(0);
        CHECK_EQ(N1, N2) << "\nTensor(" << input(0).name() << "): "
            << input(0).dim_string() << " can not Dot with Tensor"
            << "(" << input(1).name() << "): " << input(1).dim_string();
        if (input(0).template IsType<float>()) GemvRunWithType<float>();
#ifdef WITH_CUDA_FP16
        else if (input(0).template IsType<float16>()) GemvRunWithType<float16>();
#endif
        else LOG(FATAL) << "Unsupported input types.";
    } 
    else {
        LOG(FATAL) << "\nTensor(" << input(0).name() << "): "
            << input(0).dim_string() << " can not Dot with Tensor"
            << "(" << input(1).name() << "): " << input(1).dim_string();
    }
}

template <class Context>
void DotGradientOp<Context>::ShareGradient() {
    for (int i = 0; i < OutputSize(); i++) {
        if (output(i)->name() != "ignore") {
            Tensor* dX = ws()->GetBuffer("Grad");
            ws()->CreateAvatar(output(i), dX);
            break;
        }
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