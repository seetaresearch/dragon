#include "core/workspace.h"
#include "utils/filler.h"
#include "operators/arithmetic/fully_connected_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGUMENTS(X) \
    axis = OperatorBase::Arg<int64_t>("axis", 1); \
    axis = axis < 0 ? axis + X.ndim() : axis; \
    CHECK(axis >= 0 && axis < X.ndim()) \
       << "\nExcepted the axis in [-" << X.ndim() << ", " << X.ndim() \
       << "), got " << OperatorBase::Arg<int64_t>("axis", 1) << ".";

template <class Context> template <typename T>
void FullyConnectedOp<Context>::TransRunWithType() {
    vector<int64_t> W_shape({ N, K });
    vector<int64_t> bias_shape({ N });
    TENSOR_FILL(Input(1), W_shape);
    if (InputSize() > 2) TENSOR_FILL(Input(2), bias_shape);

    CHECK(Input(1).ndim() == 2 && Input(1).dim(1) == K)
        << "\nWeights dimensions should be [N, K].\n"
        << "Got X as (" << M << ", " << K << "), "
        << "and W as " << Input(1).DimString();

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Wdata = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    math::Gemm(
        CblasNoTrans, CblasTrans,
            M, N, K,
                1.f, Xdata, Wdata,
                    0.f, Ydata, ctx());

    if (InputSize() > 2) {
        DECLARE_MULTIPLIER(multiplier, M);
        auto* Bdata = Input(2).template data<T, Context>();
        math::Gemm(
            CblasNoTrans, CblasNoTrans,
                M, N, 1,
                    1.f, multiplier, Bdata,
                        1.f, Ydata, ctx());
    }
}

template <class Context> template <typename T>
void FullyConnectedOp<Context>::NoTransRunWithType() {
    vector<int64_t> W_shape({ K, N });
    vector<int64_t> bias_shape({ N });
    TENSOR_FILL(Input(1), W_shape);
    if (InputSize() > 2) TENSOR_FILL(Input(2), bias_shape);

    CHECK(Input(1).ndim() == 2 && Input(1).dim(0) == K)
        << "\nWeights dimensions should be [K, N].\n"
        << "Got X as (" << M << ", " << K << "), "
        << "and W as " << Input(1).DimString();

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Wdata = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    math::Gemm(
        CblasNoTrans, CblasNoTrans,
            M, N, K,
                1.f, Xdata, Wdata,
                    0.f, Ydata, ctx());

    if (InputSize() > 2) {
        DECLARE_MULTIPLIER(multiplier, M);
        auto* Bdata = Input(2).template data<T, Context>();
        math::Gemm(
            CblasNoTrans, CblasNoTrans,
                M, N, 1,
                    1.f, multiplier, Bdata,
                        1.f, Ydata, ctx());
    }
}

template <class Context>
void FullyConnectedOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));
    M = Input(0).count(0, axis), K = Input(0).count(axis);

    if (N <= 0) {
        // Infer the "N" from the weights shape
        N = Input(1).count() / K;
        CHECK_GT(N, 0)
            << "\nFailed to infer the N from "
            << "the weights shape: " << Input(1).DimString();
    }

    vector<int64_t> output_dims(axis + 1);
    for (int i = 0; i < axis + 1; i++) {
        output_dims[i] = i < axis ? Input(0).dim(i) : N;
    }

    Output(0)->Reshape(output_dims);

    if (XIsType(Input(0), float16)) {
        if (transW) TransRunWithType<float16>();
        else NoTransRunWithType<float16>();
    } else if (XIsType(Input(0), float)) {
        if (transW) TransRunWithType<float>();
        else NoTransRunWithType<float>();
    } else if (XIsType(Input(0), double)) {
        if (transW) TransRunWithType<double>();
        else NoTransRunWithType<double>();
    } else LOG(FATAL) << DTypeHelper(Input(0), {
        "float16",  "float32", "float64"});
}

DEPLOY_CPU(FullyConnected);
#ifdef WITH_CUDA
DEPLOY_CUDA(FullyConnected);
#endif
OPERATOR_SCHEMA(FullyConnected).NumInputs(2, 3).NumOutputs(1);

template <class Context> template <typename T>
void FullyConnectedGradientOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Wdata = Input(1).template data<T, Context>();
    auto* dYdata = Input(2).template data<T, Context>();

    if (Output(1)->name() != "NULL") {
        Output(1)->ReshapeLike(Input(1));
        auto* dWdata = Output(1)->template mutable_data<T, Context>();
        if (transW) {
            math::Gemm(
                CblasTrans, CblasNoTrans,
                    N, K, M,
                        1.f, dYdata, Xdata,
                            0.f, dWdata, ctx());
        } else {
            math::Gemm(
                CblasTrans, CblasNoTrans,
                    K, N, M,
                        1.f, Xdata, dYdata,
                            0.f, dWdata, ctx());
        }
    }

    if (Output(2)->name() != "NULL") {
        DECLARE_MULTIPLIER(multiplier, M);
        Output(2)->Reshape({ N });
        auto* dBdata = Output(2)->template mutable_data<T, Context>();
        math::Gemv(
            CblasTrans, M, N,
                1.f, dYdata, multiplier,
                    0.f, dBdata, ctx());
    }

    if (Output(0)->name() != "NULL") {
        Output(0)->ReshapeLike(Input(0));
        auto* dXdata = Output(0)->template mutable_data<T, Context>();
        if (transW) {
            math::Gemm(
                CblasNoTrans, CblasNoTrans,
                    M, K, N,
                        1.f, dYdata, Wdata,
                            0.f, dXdata, ctx());
        } else {
            math::Gemm(
                CblasNoTrans, CblasTrans,
                    M, K, N,
                        1.f, dYdata, Wdata,
                            0.f, dXdata, ctx());
        }
    }
}

template <class Context>
void FullyConnectedGradientOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));
    M = Input(0).count(0, axis), K = Input(0).count(axis);

    if (N <= 0) {
        // Infer the "N" from the weights shape
        N = Input(1).count() / K;
        CHECK_GT(N, 0)
            << "\nFailed to infer the N from "
            << "the weights shape: " << Input(1).DimString();
    }

    if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0), {
        "float16", "float32", "float64" });
}

DEPLOY_CPU(FullyConnectedGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(FullyConnectedGradient);
#endif

OPERATOR_SCHEMA(FullyConnectedGradient)
    .NumInputs(3).NumOutputs(3);

class GetFullyConnectedGradient : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetFullyConnectedGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0), GI(1), GI(2) }));
    }
};

REGISTER_GRADIENT(FullyConnected, GetFullyConnectedGradient);

#undef DETERMINE_RUNTIME_ARGUMENTS

}  // namespace dragon