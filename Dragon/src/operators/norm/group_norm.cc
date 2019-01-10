#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/norm/group_norm_op.h"

namespace dragon {

template <class Context> template <typename Tx, typename Tp>
void GroupNormOp<Context>::RunWithType() {
    TENSOR_FILL_WITH_TYPE(Input(1), vector<int64_t>({ C }), Tp);
    TENSOR_FILL_WITH_TYPE(Input(2), vector<int64_t>({ C }), Tp);

    auto* x = Input(0).template data<Tx, Context>();
    auto* gamma = Input(1).template data<Tp, Context>();
    auto* beta = Input(2).template data<Tp, Context>();
    auto* mu = mean->template mutable_data<Tp, Context>();
    auto* rsig = var->template mutable_data<Tp, Context>();
    auto* s = scale.template mutable_data<Tp, Context>();
    auto* b = bias.template mutable_data<Tp, Context>();
    auto* y = Output(0)->template mutable_data<Tx, Context>();

    // Compute the moments
    if (data_format == "NCHW") {
        vector<int> dims = { (int)(N * G), (int)(D * S) };
        vector<int> axes = { 1 };
        kernel::Moments(
            2, dims.data(), 1, axes.data(),
                x, mu, rsig, ctx());
    } else if (data_format == "NHWC") {
        vector<int> dims = { (int)N, (int)S, (int)G, (int)D };
        vector<int> axes = { 1, 3 };
        kernel::Moments(
            4, dims.data(), 2, axes.data(),
                x, mu, rsig, ctx());
    }

    math::InvStd(N * G, eps, rsig, rsig, ctx());
    kernel::GroupNormForward(N, G, D, S, data_format,
        x, mu, rsig, gamma, beta, s, b, y, ctx());
}

template <class Context>
void GroupNormOp<Context>::Reshape() {
    // Determine the data format
    int64_t channel_axis = axis;
    data_format = "NCHW";
    if (channel_axis == -1) channel_axis += Input(0).ndim();
    if (channel_axis + 1 == Input(0).ndim()) data_format = "NHWC";
    if (Input(0).ndim() == 2) data_format = "NCHW";
    N = Input(0).dim(0); C = Input(0).dim(channel_axis);
    S = Input(0).count() / N / C;

    // InstanceNorm, LayerNorm or GroupNorm ?
    G = group > 0 ? group : C; D = C / G;

    // Check the channels and groups
    CHECK_EQ(C % G, 0) << "\nThe " << C << " channels "
        << "can not be split into " << G << " groups.";
    if (G == C && Input(0).ndim() == 2)
        LOG(WARNING) << "The 2d input will output all zeros.";

    // Create the shared resources
    mean = ws()->CreateTensor(mount_name(
        "gn/mu"))->Reshape({ N * G });
    var = ws()->CreateTensor(mount_name(
        "gn/rsig"))->Reshape({ N * G });

    // Reshape
    scale.Reshape({ N * C }); bias.Reshape({ N * C });
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void GroupNormOp<Context>::RunOnDevice() {
    Reshape();

    if (XIsType(Input(0), float)) RunWithType<float, float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16, float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(GroupNorm);
#ifdef WITH_CUDA
DEPLOY_CUDA(GroupNorm);
#endif

OPERATOR_SCHEMA(GroupNorm)
    .NumInputs(3).NumOutputs(1);

template <class Context> template <typename Tx, typename Tp>
void GroupNormGradientOp<Context>::RunWithType() {
    auto* x = Input(0).template data<Tx, Context>();
    auto* mu = mean->template data<Tp, Context>();
    auto* rsig = var->template data<Tp, Context>();
    auto* gamma = Input(1).template data<Tp, Context>();
    auto* dy = Input(-1).template data<Tx, Context>();
    auto* ds = dscale.template mutable_data<Tp, Context>();
    auto* db = dbias.template mutable_data<Tp, Context>();
    auto* dx = Output(0)->template mutable_data<Tx, Context>();
    auto* dgamma = Output(1)->template mutable_data<Tp, Context>();
    auto* dbeta = Output(2)->template mutable_data<Tp, Context>();

    kernel::GroupNormBackward(
        N, G, D, S, data_format,
            x, mu, rsig, gamma, dy,
                ds, db, dx, dgamma, dbeta, ctx());
}

template <class Context>
void GroupNormGradientOp<Context>::Reshape() {
    // Determine the data format
    int64_t channel_axis = axis;
    data_format = "NCHW";
    if (channel_axis == -1) channel_axis += Input(0).ndim();
    if (channel_axis + 1 == Input(0).ndim()) data_format = "NHWC";
    if (Input(0).ndim() == 2) data_format = "NCHW";
    N = Input(0).dim(0); C = Input(0).dim(channel_axis);
    S = Input(0).count() / N / C;

    // InstanceNorm, LayerNorm or GroupNorm ?
    G = group > 0 ? group : C; D = C / G;

    // Check the channels and groups
    CHECK_EQ(C % G, 0) << "\nThe " << C << " channels "
        << "can not be split into " << G << " groups.";
    if (G == C && Input(0).ndim() == 2)
        LOG(WARNING) << "The 2d input will output all zeros.";

    // Get the shared resources
    mean = ws()->GetTensor(mount_name("gn/mu"));
    var = ws()->GetTensor(mount_name("gn/rsig"));

    // Reshape
    dscale.Reshape({ N * G }); dbias.Reshape({ N * G });
    Output(0)->ReshapeLike(Input(0));  // dx
    Output(1)->Reshape({ C });         // dgamma
    Output(2)->Reshape({ C });         // dbeta
}

template <class Context>
void GroupNormGradientOp<Context>::RunOnDevice() {
    Reshape();

    if (XIsType(Input(0), float)) RunWithType<float, float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16, float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(GroupNormGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(GroupNormGradient);
#endif

OPERATOR_SCHEMA(GroupNormGradient)
    .NumInputs(3).NumOutputs(3);

class GetGroupNormGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetGroupNormGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0), GI(1), GI(2) }));
    }
};

REGISTER_GRADIENT(GroupNorm, GetGroupNormGradient);

}  // namespace dragon