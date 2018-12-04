#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/math_functions.h"
#include "operators/norm/group_norm_op.h"

namespace dragon {

template <class Context> template <typename T>
void GroupNormOp<Context>::RunWithType() {
    DECLARE_MULTIPLIER(MXmult, std::max(NS, CGS));

    auto* Tmean = mean.template mutable_data<T, Context>();
    auto* Tvar = var->template mutable_data<T, Context>();
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    auto* NCdata = nc.template mutable_data<T, Context>();
    auto* WSdata = ws()->template caches<T, Context>({ Input(0).count() })[0];
    ctx()->template Copy<T, Context, Context>(Output(0)->count(), Ydata, Xdata);

    // Compute mean
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(
            CblasNoTrans, NG, CGS,
                1.f / CGS, Xdata, MXmult,
                    0.f, Tmean, ctx());
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }

    // Subtract mean
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NG, CGS, 1,
                    -1.f, Tmean, MXmult,
                        1.f, Ydata, ctx());
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }

    // Compute variance
    // Note that we use VAR(X) = E((X - EX) ^ 2)
    math::Square<T, Context>(Output(0)->count(), Ydata, WSdata, ctx());
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(
            CblasNoTrans, NG, CGS,
                1.f / CGS, WSdata, MXmult,
                    0.f, Tvar, ctx());
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }

    // Compute stddev
    math::AddScalar<T, Context>(var->count(), eps, Tvar, ctx());
    math::Sqrt<T, Context>(var->count(), Tvar, Tvar, ctx());

    // Divide by stddev
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NG, CGS, 1,
                    1.f, Tvar, MXmult,
                        0.f, WSdata, ctx());
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }
    math::Div<T, Context>(Output(0)->count(),
        Ydata, WSdata, Ydata, ctx());
}

template <class Context>
void GroupNormOp<Context>::Setup() {
    // Determine the data format
    TIndex channel_axis = axis;
    data_format = "NCHW";
    if (channel_axis == -1) channel_axis += (int)Input(0).ndim();
    if (channel_axis + 1 == (int)Input(0).ndim()) data_format = "NHWC";
    if (Input(0).ndim() == 2) data_format = "NCHW";
    N = Input(0).dim(0);
    C = Input(0).dim(channel_axis);
    CHECK_EQ(C % group, 0) << "\nThe " << C << " channels "
        << "can not be split into " << group << " groups.";
    if (group == C && Input(0).ndim() == 2)  // InstanceNorm
        LOG(WARNING) << "The 2d input will output all zeros.";
    NC = N * C;
    NG = N * group;
    S = Input(0).count() / NC;
    CGS = (C / group) * S;
    NS = N * S;

    // Make resource
    var = ws()->CreateTensor("/mnt/" + anchor() + "/gn/var");

    // Reshape
    mean.Reshape({ NG });
    var->Reshape({ NG });
    nc.Reshape({ NC });
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void GroupNormOp<Context>::RunOnDevice() {
    Setup();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(GroupNorm);
#ifdef WITH_CUDA
DEPLOY_CUDA(GroupNorm);
#endif
OPERATOR_SCHEMA(GroupNorm).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void GroupNormGradientOp<Context>::RunWithType() {
    DECLARE_MULTIPLIER(MXmult, std::max(NS, CGS));

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Tvar = var->template mutable_data<T, Context>();
    auto* NCdata = nc.template mutable_data<T, Context>();
    auto* WSdata = ws()->template caches<T, Context>({ Output(0)->count() })[0];

    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NG, CGS, 1,
                    1.f, Tvar, MXmult,
                        0.f, WSdata, ctx());
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }

    auto* Ydata = Input(1).template data<T, Context>();
    math::Mul<T, Context>(Output(0)->count(),
        Ydata, dYdata, dXdata, ctx());

     // sum(dE/dY \cdot Y)
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(
            CblasNoTrans, NG, CGS,
                1.f, dXdata, MXmult,
                    0.f, Tvar, ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NG, CGS, 1,
                    1.f, Tvar, MXmult,
                        0.f, dXdata, ctx());
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }

    // sum(dE/dY \cdot Y) \cdot Y
    math::Mul<T, Context>(Output(0)->count(),
        Ydata, dXdata, dXdata, ctx());

    // sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(
            CblasNoTrans, NG, CGS,
                1.f, dYdata, MXmult,
                    0.f, Tvar, ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NG, CGS, 1,
                    1.f, Tvar, MXmult,
                        1.f, dXdata, ctx());
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }

    //   dE/dY - mean(dE/dY)- mean(dE/dY \cdot Y) \cdot Y
    // = dE/dY - mean(sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y)
    math::Axpby<T, Context>(Output(0)->count(),
        1.f, dYdata, -1.f / CGS, dXdata, ctx());

    // Divide by stddev
    math::Div<T, Context>(Output(0)->count(),
        dXdata, WSdata, dXdata, ctx());
}

template <class Context>
void GroupNormGradientOp<Context>::Setup() {
    // Determine the data format
    TIndex channel_axis = axis;
    data_format = "NCHW";
    if (channel_axis == -1) channel_axis += (int)Input(0).ndim();
    if (channel_axis + 1 == (int)Input(0).ndim()) data_format = "NHWC";
    if (Input(0).ndim() == 2) data_format = "NCHW";
    N = Input(0).dim(0);
    C = Input(0).dim(channel_axis);
    CHECK_EQ(C % group, 0) << "\nThe " << C << " channels "
        << "can not be split into " << group << " groups.";
    if (group == C && Input(0).ndim() == 2)  // InstanceNorm
        LOG(WARNING) << "The 2d input will output all zeros.";
    NC = N * C;
    NG = N * group;
    S = Input(0).count() / NC;
    CGS = (C / group) * S;
    NS = N * S;

    // Make resource
    var = ws()->GetTensor("/mnt/" + anchor() + "/gn/var");

    // Reshape
    nc.Reshape({ NC });
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void GroupNormGradientOp<Context>::RunOnDevice() {
    Setup();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(GroupNormGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(GroupNormGradient);
#endif
OPERATOR_SCHEMA(GroupNormGradient).NumInputs(3).NumOutputs(1);

class GetGroupNormGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetGroupNormGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), O(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(GroupNorm, GetGroupNormGradient);

}  // namespace dragon