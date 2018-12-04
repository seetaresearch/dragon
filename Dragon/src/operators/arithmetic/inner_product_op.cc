#include "core/workspace.h"
#include "utils/filler.h"
#include "operators/arithmetic/inner_product_op.h"

namespace dragon {

template <class Context> template <typename T>
void InnerProductOp<Context>::TransRunWithType() {
    vector<TIndex> weight_shape({ num_output, K });
    vector<TIndex> bias_shape(1, num_output);
    TENSOR_FILL(Input(1), weight_shape);
    if (InputSize() > 2) TENSOR_FILL(Input(2), bias_shape);
    CHECK(Input(1).ndim() == 2 && Input(1).dim(1) == K)
        << "\nWeights should shape as [num_output, dim].\n"
        << "Input dims are (" << M << ", " << K << ").\n"
        << "Weights dims are " << Input(1).DimString();

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Wdata = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    math::Gemm<T, Context>(
        CblasNoTrans, CblasTrans,
            M, num_output, K,
                1.f, Xdata, Wdata,
                    0.f, Ydata, ctx());

    if (InputSize() > 2) {
        DECLARE_MULTIPLIER(multiplier, M);
        auto* Bdata = Input(2).template data<T, Context>();
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                M, num_output, 1,
                    1.f, multiplier, Bdata,
                        1.f, Ydata, ctx());
    }
}

template <class Context> template <typename T>
void InnerProductOp<Context>::NoTransRunWithType() {
    vector<TIndex> weight_shape({ K, num_output });
    vector<TIndex> bias_shape(1, num_output);
    TENSOR_FILL(Input(1), weight_shape);
    if (InputSize() > 2) TENSOR_FILL(Input(2), bias_shape);
    CHECK(Input(1).ndim() == 2 && Input(1).dim(0) == K)
        << "\nWeights should shape as [num_output, dim].\n"
        << "Input dims are (" << M << ", " << K << ").\n"
        << "Weights dims are " << Input(1).DimString();

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Wdata = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    math::Gemm<T, Context>(
        CblasNoTrans, CblasNoTrans,
            M, num_output, K,
                1.f, Xdata, Wdata,
                    0.f, Ydata, ctx());

    if (InputSize() > 2) {
        DECLARE_MULTIPLIER(multiplier, M);
        auto* Bdata = Input(2).template data<T, Context>();
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                M, num_output, 1,
                    1.f, multiplier, Bdata,
                        1.f, Ydata, ctx());
    }
}

template <class Context>
void InnerProductOp<Context>::RunOnDevice() {
    TIndex _axis_ = axis < 0 ? axis + Input(0).ndim() : axis;
    M = Input(0).count(0, _axis_), K = Input(0).count(_axis_);

    vector<TIndex> output_dims(_axis_ + 1);
    for (int i = 0; i < _axis_ + 1; i++)
        output_dims[i] = i < _axis_ ?
            Input(0).dim(i) : num_output;
    Output(0)->Reshape(output_dims);

    if (XIsType(Input(0), float)) {
        if (TransW) TransRunWithType<float>();
        else NoTransRunWithType<float>();
    } else if (XIsType(Input(0), float16)) {
        if (TransW) TransRunWithType<float16>();
        else NoTransRunWithType<float16>();
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(InnerProduct);
#ifdef WITH_CUDA
DEPLOY_CUDA(InnerProduct);
#endif
OPERATOR_SCHEMA(InnerProduct).NumInputs(2, 3).NumOutputs(1);

template <class Context> template <typename T>
void InnerProductGradientOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Wdata = Input(1).template data<T, Context>();
    auto* dYdata = Input(2).template data<T, Context>();

    if (Output(1)->name() != "ignore") {
        Output(1)->ReshapeLike(Input(1));
        auto* dWdata = Output(1)->template mutable_data<T, Context>(ctx());
        if (TransW) {
            math::Gemm<T, Context>(
                CblasTrans, CblasNoTrans,
                    num_output, K, M,
                        1.f, dYdata, Xdata,
                            1.f, dWdata, ctx());
        } else {
            math::Gemm<T, Context>(
                CblasTrans, CblasNoTrans,
                    K, num_output, M,
                        1.f, Xdata, dYdata,
                            1.f, dWdata, ctx());
        }
    }

    if (Output(2)->name() != "ignore") {
        DECLARE_MULTIPLIER(multiplier, M);
        Output(2)->Reshape({ num_output });
        auto* dBdata = Output(2)->template mutable_data<T, Context>(ctx());
        math::Gemv<T, Context>(
            CblasTrans, M, num_output,
                1.f, dYdata, multiplier,
                    1.f, dBdata, ctx());
    }

    if (Output(0)->name() != "ignore") {
        Output(0)->ReshapeLike(Input(0));
        auto* dXdata = Output(0)->template mutable_data<T, Context>();
        if (TransW) {
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    M, K, num_output,
                        1.f, dYdata, Wdata,
                            0.f, dXdata, ctx());
        } else {
            math::Gemm<T, Context>(
                CblasNoTrans, CblasTrans,
                    M, K, num_output,
                        1.f, dYdata, Wdata,
                            0.f, dXdata, ctx());
        }
    }
}

template <class Context>
void InnerProductGradientOp<Context>::RunOnDevice() {
    TIndex _axis_ = axis < 0 ? axis + Input(0).ndim() : axis;
    M = Input(0).count(0, _axis_), K = Input(0).count(_axis_);

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(InnerProductGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(InnerProductGradient);
#endif
OPERATOR_SCHEMA(InnerProductGradient).NumInputs(3).NumOutputs(3);

class GetInnerProductGradient : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetInnerProductGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0), GI(1), GI(2)});
    }
};
REGISTER_GRADIENT(InnerProduct, GetInnerProductGradient);

}  // namespace dragon