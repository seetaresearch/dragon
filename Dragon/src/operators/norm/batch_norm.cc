#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/norm/batch_norm_op.h"

namespace dragon {

template <class Context> template <typename Tx, typename Tp>
void BatchNormOp<Context>::TrainingRunWithType() {
    TENSOR_FILL_WITH_TYPE(Input(1), vector<int64_t>({ C }), Tp);
    TENSOR_FILL_WITH_TYPE(Input(2), vector<int64_t>({ C }), Tp);
    TENSOR_FILL_WITH_TYPE(Input(3), vector<int64_t>({ C }), Tp);
    TENSOR_FILL_WITH_TYPE(Input(4), vector<int64_t>({ C }), Tp);

    auto* x = Input(0).template data<Tx, Context>();
    auto* rm = Input(1).template mutable_data<Tp, Context>();
    auto* rv = Input(2).template mutable_data<Tp, Context>();
    auto* gamma = Input(3).template data<Tp, Context>();
    auto* beta = Input(4).template data<Tp, Context>();
    auto* mu = mean->template mutable_data<Tp, Context>();
    auto* rsig = var->template mutable_data<Tp, Context>();
    auto* s = scale.template mutable_data<Tp, Context>();
    auto* b = bias.template mutable_data<Tp, Context>();
    auto* y = Output(0)->template mutable_data<Tx, Context>();

    // Compute the moments
    if (data_format == "NCHW") {
        const std::array<int, 3> dims = { (int)N, (int)C, (int)S };
        const std::array<int, 3> axes = { 0, 2 };
        kernel::Moments(
            3, dims.data(), 2, axes.data(),
                x, mu, rsig, ctx());
    } else if (data_format == "NHWC") {
        const std::array<int, 2> dims = { (int)(N * S), (int)C };
        const std::array<int, 1> axes = { 0 };
        kernel::Moments(
            2, dims.data(), 1, axes.data(),
                x, mu, rsig, ctx());
    }

    // Compute moving average
    if (!is_recomputing) {
        // Running(X) = (1 - momentum) * Cur(X) + momentum * Running(X)
        math::Axpby<Tp, Context>(mean->count(),
            1.f - momentum, mu, momentum, rm, ctx());
        math::Axpby<Tp, Context>(var->count(),
            1.f - momentum, rsig, momentum, rv, ctx());
    }

    // Fuse: [mu, rsig, alpha, beta] => [scale, bias]
    math::InvStd(C, eps, rsig, rsig, ctx());
    math::Mul(C, gamma, rsig, s, ctx());
    math::Mul(C, s, mu, b, ctx());
    math::Sub(C, beta, b, b, ctx());

    // Affine
    if (data_format == "NCHW") {
        kernel::Affine(N, S, C, x, s, b, y, ctx());
    } else if (data_format == "NHWC") {
        kernel::Affine(N * S, 1, C, x, s, b, y, ctx());
    }
}

template <class Context> template <typename Tx, typename Tp>
void BatchNormOp<Context>::InferenceRunWithType() {
    TENSOR_FILL_WITH_TYPE(Input(1), vector<int64_t>({ C }), Tp);
    TENSOR_FILL_WITH_TYPE(Input(2), vector<int64_t>({ C }), Tp);
    TENSOR_FILL_WITH_TYPE(Input(3), vector<int64_t>({ C }), Tp);
    TENSOR_FILL_WITH_TYPE(Input(4), vector<int64_t>({ C }), Tp);

    auto* x = Input(0).template data<Tx, Context>();
    auto* rm = Input(1).template data<Tp, Context>();
    auto* rv = Input(2).template data<Tp, Context>();
    auto* gamma = Input(3).template data<Tp, Context>();
    auto* beta = Input(4).template data<Tp, Context>();
    auto* s = scale.template mutable_data<Tp, Context>();
    auto* b = bias.template mutable_data<Tp, Context>();
    auto* y = Output(0)->template mutable_data<Tx, Context>();

    // Fuse: [rmean, rvar, alpha, beta] => [scale, bias]
    math::InvStd(C, eps, rv, b, ctx());
    math::Mul(C, gamma, b, s, ctx());
    math::Mul(C, s, rm, b, ctx());
    math::Sub(C, beta, b, b, ctx());

    // Affine
    if (data_format == "NCHW") {
        kernel::Affine(N, S, C, x, s, b, y, ctx());
    } else if (data_format == "NHWC") {
        kernel::Affine(N * S, 1, C, x, s, b, y, ctx());
    }
}

template <class Context>
void BatchNormOp<Context>::Reshape() {
    // Determine the mode
    if (use_stats == -1) {
        is_training = phase() == "TRAIN" ? true : false;
    } else {
        is_training = use_stats == 0 ? true : false;
    }

    is_recomputing = ws()->GetTensor(
        "/opt/recomputing_flag")->template
            data<bool, CPUContext>()[0];

    // Determine the data format
    int64_t channel_axis = axis;
    data_format = "NCHW";
    if (channel_axis == -1) channel_axis += Input(0).ndim();
    if (channel_axis + 1 == Input(0).ndim()) data_format = "NHWC";
    N = Input(0).dim(0); C = Input(0).dim(channel_axis);
    S = Input(0).count() / N / C;

    // Create the shared resources
    mean = ws()->CreateTensor(mount_name(
        "bn/mu"))->Reshape({ C });
    var = ws()->CreateTensor(mount_name(
        "bn/rsig"))->Reshape({ C });

    // Reshape
    scale.Reshape({ C }); bias.Reshape({ C });
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void BatchNormOp<Context>::RunOnDevice() {
    Reshape();

    if (XIsType(Input(0), float)) {
        if (is_training) TrainingRunWithType<float, float>();
        else InferenceRunWithType<float, float>();
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(BatchNorm);
#ifdef WITH_CUDA
DEPLOY_CUDA(BatchNorm);
#endif

OPERATOR_SCHEMA(BatchNorm)
    .NumInputs(5).NumOutputs(1);

template <class Context> template <typename Tx, typename Tp>
void BatchNormGradientOp<Context>::TrainingRunWithType() {
    auto* x = Input(0).template data<Tx, Context>();
    auto* mu = mean->template data<Tp, Context>();
    auto* rsig = var->template data<Tp, Context>();
    auto* gamma = Input(3).template data<Tp, Context>();
    auto* dy = Input(-1).template data<Tx, Context>();
    auto* ds = dscale.template mutable_data<Tp, Context>();
    auto* db = dbias.template mutable_data<Tp, Context>();
    auto* dx = Output(0)->template mutable_data<Tx, Context>();
    auto* dgamma = Output(1)->template mutable_data<Tp, Context>();
    auto* dbeta = Output(2)->template mutable_data<Tp, Context>();

    kernel::BatchNormBackwardTraining(
        N, C, S, data_format,
            x, mu, rsig, gamma, dy,
                ds, db, dx, dgamma, dbeta, ctx());
}

template <class Context> template <typename Tx, typename Tp>
void BatchNormGradientOp<Context>::InferenceRunWithType() {
    auto* x = Input(0).template data<Tx, Context>();
    auto* rm = Input(1).template data<Tp, Context>();
    auto* rv = Input(2).template data<Tp, Context>();
    auto* gamma = Input(3).template data<Tp, Context>();
    auto* dy = Input(-1).template data<Tx, Context>();
    auto* dx = Output(0)->template mutable_data<Tx, Context>();
    auto* rsig = var->template mutable_data<Tp, Context>();

    Tp* dgamma = nullptr, *dbeta = nullptr;

    // Gradient w.r.t. gamma or beta if necessary
    if (Output(1)->name() != "NULL" ||
            Output(2)->name() != "NULL") {
        dgamma = Output(1)->template mutable_data<Tp, Context>();
        dbeta = Output(2)->template mutable_data<Tp, Context>();
    }

    math::InvStd(C, eps, rv, rsig, ctx());

    kernel::BatchNormBackwardInference(
        N, C, S, data_format,
            x, rm, rsig, gamma, dy,
                dx, dgamma, dbeta, ctx());
}

template <class Context>
void BatchNormGradientOp<Context>::Reshape() {
    // Determine the mode
    if (use_stats == -1) {
        is_training = phase() == "TRAIN" ? true : false;
    } else {
        is_training = use_stats == 0 ? true : false;
    }

    // Determine the data format
    int64_t channel_axis = axis;
    data_format = "NCHW";
    if (channel_axis == -1) channel_axis += Input(0).ndim();
    if (channel_axis + 1 == Input(0).ndim()) data_format = "NHWC";
    N = Input(0).dim(0); C = Input(0).dim(channel_axis);
    S = Input(0).count() / N / C;

    // Get the shared resources
    mean = ws()->GetTensor(mount_name("bn/mu"));
    var = ws()->GetTensor(mount_name("bn/rsig"));

    // Reshape
    dscale.Reshape({ C }); dbias.Reshape({ C });
    Output(0)->ReshapeLike(Input(0));  // dx
    Output(1)->Reshape({ C });         // dgamma
    Output(2)->Reshape({ C });         // dbeta
}

template <class Context>
void BatchNormGradientOp<Context>::RunOnDevice() {
    Reshape();

    if (XIsType(Input(0), float)) {
        if (is_training) TrainingRunWithType<float, float>();
        else InferenceRunWithType<float, float>();
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(BatchNormGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(BatchNormGradient);
#endif

OPERATOR_SCHEMA(BatchNormGradient)
    .NumInputs(5).NumOutputs(3);

class GetBatchNormGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetBatchNormGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), I(2), I(3), GO(0) }),
            vector<string>({ GI(0), GI(3), GI(4) }));
    }
};

REGISTER_GRADIENT(BatchNorm, GetBatchNormGradient);

}  // namespace dragon