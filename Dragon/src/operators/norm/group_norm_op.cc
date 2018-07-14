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
    ctx().template Copy<T, Context, Context>(Output(0)->count(), Ydata, Xdata);

    //  compute mean
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(
            CblasNoTrans, NG, CGS,
                1.0 / CGS, Xdata, MXmult,
                    0, Tmean, &ctx());
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }

    //  subtract mean
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NG, CGS, 1,
                    -1.0, Tmean, MXmult,
                        1.0, Ydata, &ctx());
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }

    //  compute variance
    //  note that we use VAR(X) = E((X - EX) ^ 2)
    math::Square<T, Context>(Output(0)->count(), Ydata, WSdata);
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(
            CblasNoTrans, NG, CGS,
                1.0 / CGS, WSdata, MXmult,
                    0.0, Tvar, &ctx());
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }

    //  compute stddev
    math::AddScalar<T, Context>(var->count(), eps, Tvar);
    math::Sqrt<T, Context>(var->count(), Tvar, Tvar);

    //  divide by stddev
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NG, CGS, 1,
                    1.0, Tvar, MXmult,
                        0.0, WSdata, &ctx());
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }
    math::Div<T, Context>(Output(0)->count(), Ydata, WSdata, Ydata);
}

template <class Context>
void GroupNormOp<Context>::Setup() {
    //  determine the data format
    TIndex channel_axis = axis;
    data_format = "NCHW";
    if (channel_axis == -1) channel_axis += (int)Input(0).ndim();
    if (channel_axis + 1 == (int)Input(0).ndim()) data_format = "NHWC";
    if (Input(0).ndim() == 2) data_format = "NCHW";
    N = Input(0).dim(0);
    C = Input(0).dim(channel_axis);
    CHECK_EQ(C % group, 0) << "\nThe " << C << " channels "
        << "can not be split into " << group << " groups.";
    if (group == C && Input(0).ndim() == 2)  //  InstanceNorm
        LOG(WARNING) << "The 2d input will output all zeros.";
    NC = N * C;
    NG = N * group;
    S = Input(0).count() / NC;
    CGS = (C / group) * S;
    NS = N * S;

    //  make resource
    var = ws()->CreateTensor("/mnt/" + anchor() + "/gn/var");

    //  reshape
    mean.Reshape({ NG });
    var->Reshape({ NG });
    nc.Reshape({ NC });
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void GroupNormOp<Context>::RunOnDevice() {
    Setup();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
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
                    1.0, Tvar, MXmult,
                        0.0, WSdata, &ctx());
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }

    auto* Ydata = Input(1).template data<T, Context>();
    math::Mul<T, Context>(Output(0)->count(), Ydata, dYdata, dXdata);

     //  sum(dE/dY \cdot Y)
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(
            CblasNoTrans, NG, CGS,
                1.0, dXdata, MXmult,
                    0.0, Tvar, &ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NG, CGS, 1,
                    1.0, Tvar, MXmult,
                        0.0, dXdata, &ctx());
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }

    //  sum(dE/dY \cdot Y) \cdot Y
    math::Mul<T, Context>(Output(0)->count(), Ydata, dXdata, dXdata);

    //  sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(
            CblasNoTrans, NG, CGS,
                1.0, dYdata, MXmult,
                    0.0, Tvar, &ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NG, CGS, 1,
                    1.0, Tvar, MXmult,
                        1.0, dXdata, &ctx());
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }

    //   dE/dY - mean(dE/dY)- mean(dE/dY \cdot Y) \cdot Y
    // = dE/dY - mean(sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y)
    math::Axpby<T, Context>(Output(0)->count(),
        1.0, dYdata, -1.0 / CGS, dXdata, &ctx());

    //  divide by stddev
    math::Div<T, Context>(Output(0)->count(), dXdata, WSdata, dXdata);
}

template <class Context>
void GroupNormGradientOp<Context>::Setup() {
    //  determine the data format
    TIndex channel_axis = axis;
    data_format = "NCHW";
    if (channel_axis == -1) channel_axis += (int)Input(0).ndim();
    if (channel_axis + 1 == (int)Input(0).ndim()) data_format = "NHWC";
    if (Input(0).ndim() == 2) data_format = "NCHW";
    N = Input(0).dim(0);
    C = Input(0).dim(channel_axis);
    CHECK_EQ(C % group, 0) << "\nThe " << C << " channels "
        << "can not be split into " << group << " groups.";
    if (group == C && Input(0).ndim() == 2)  //  InstanceNorm
        LOG(WARNING) << "The 2d input will output all zeros.";
    NC = N * C;
    NG = N * group;
    S = Input(0).count() / NC;
    CGS = (C / group) * S;
    NS = N * S;

    //  make resource
    var = ws()->GetTensor("/mnt/" + anchor() + "/gn/var");

    //  reshape
    nc.Reshape({ NC });
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void GroupNormGradientOp<Context>::RunOnDevice() {
    Setup();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
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

}    // namespace dragon