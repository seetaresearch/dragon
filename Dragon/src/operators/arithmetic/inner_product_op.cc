#include "operators/arithmetic/inner_product_op.h"
#include "core/workspace.h"
#include "utils/filler.h"

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
        << "Weights dims are " << Input(1).dim_string();
    Output(0)->Reshape(vector<TIndex>({ M, num_output }));

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Wdata = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    math::Gemm<T, Context>(
        CblasNoTrans, CblasTrans,
            M, num_output, K,
                1.0, Xdata, Wdata, 0.0, Ydata);

    if (InputSize() > 2) {
        DECLARE_MULTIPLIER(multiplier, M);
        auto* Bdata = Input(2).template data<T, Context>();
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                M, num_output, 1,
                    1.0, multiplier, Bdata, 1.0, Ydata);
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
        << "Weights dims are " << Input(1).dim_string();
    Output(0)->Reshape(vector<TIndex>({ M, num_output }));

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Wdata = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    math::Gemm<T, Context>(
        CblasNoTrans, CblasNoTrans,
            M, num_output, K,
                1.0, Xdata, Wdata, 0.0, Ydata);

    if (InputSize() > 2) {
        DECLARE_MULTIPLIER(multiplier, M);
        auto* Bdata = Input(2).template data<T, Context>();
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                M, num_output, 1,
                    1.0, multiplier, Bdata, 1.0, Ydata);
    }
}

template <class Context>
void InnerProductOp<Context>::RunOnDevice() {
    TIndex _axis_ = axis < 0 ? axis + Input(0).ndim() : axis;
    M = Input(0).count(0, _axis_), K = Input(0).count(_axis_);

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
        auto* dWdata = Output(1)->template mutable_data<T, Context>();
        if (TransW) {
            math::Gemm<T, Context>(
                CblasTrans, CblasNoTrans,
                    num_output, K, M,
                        1.0, dYdata, Xdata, 1.0, dWdata);
        } else {
            math::Gemm<T, Context>(
                CblasTrans, CblasNoTrans,
                    K, num_output, M,
                        1.0, Xdata, dYdata, 1.0, dWdata);
        }
    }

    if (Output(2)->name() != "ignore") {
        DECLARE_MULTIPLIER(multiplier, M);
        Output(2)->Reshape(vector<TIndex>(1, num_output));
        auto* dBdata = Output(2)->template mutable_data<T, Context>();
        math::Gemv<T, Context>(
            CblasTrans,
                M, num_output,
                    1.0, dYdata, multiplier, 1.0, dBdata);
    }

    if (Output(0)->name() != "ignore") {
        Output(0)->ReshapeLike(Input(0));
        auto* dXdata = Output(0)->template mutable_data<T, Context>();
        if (TransW) {
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    M, K, num_output,
                        1.0, dYdata, Wdata, 0.0, dXdata);
        } else {
            math::Gemm<T, Context>(
                CblasNoTrans, CblasTrans,
                    M, K, num_output,
                        1.0, dYdata, Wdata, 0.0, dXdata);
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

}    // namespace dragon