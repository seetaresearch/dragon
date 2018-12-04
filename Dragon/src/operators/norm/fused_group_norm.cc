#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/math_functions.h"
#include "operators/norm/group_norm_op.h"

namespace dragon {

template <class Context> template <typename T>
void FusedGroupNormOp<Context>::RunWithType() {
    TENSOR_FILL(Input(1), vector<TIndex>(1, C));  // scale
    TENSOR_FILL(Input(2), vector<TIndex>(1, C));  // bias

    DECLARE_MULTIPLIER(MXmult, std::max(NS, CGS));

    auto* Sdata = Input(1).template data<T, Context>();
    auto* Bdata = Input(2).template data<T, Context>();
    auto* Tmean = mean->template mutable_data<T, Context>();
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

    // Store x_norm for backward
    auto* XNorm_data = x_norm->template mutable_data<T, Context>();
    ctx()->template Copy<T, Context, Context>(
        Output(0)->count(), XNorm_data, Ydata);

    // Scale
    if (data_format == "NCHW") {
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans,
                 N, C, 1,
                     1.f, MXmult, Sdata,
                         0.f, NCdata, ctx());
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans,
                 NC, S, 1,
                     1.f, NCdata, MXmult,
                         0.f, WSdata, ctx());
    } else if (data_format == "NHWC") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                 NS, C, 1,
                     1.f, MXmult, Sdata,
                         0.f, WSdata, ctx());
    }
    math::Mul<T, Context>(Output(0)->count(),
        Ydata, WSdata, Ydata, ctx());

    // Shift
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                N, C, 1,
                    1.f, MXmult, Bdata,
                        0.f, NCdata, ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NC, S, 1,
                    1.f, NCdata, MXmult,
                        1.f, Ydata, ctx());
    } else if (data_format == "NHWC") {
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans,
                NS, C, 1,
                    1.f, MXmult,  Bdata,
                        1.f, Ydata, ctx());
    }
}

template <class Context>
void FusedGroupNormOp<Context>::Setup() {
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
    if (group == C && Input(0).ndim() == 2)    //  InstanceNorm
        LOG(WARNING) << "The 2d input will output all zeros.";
    NC = N * C;
    NG = N * group;
    S = Input(0).count() / NC;
    CGS = (C / group) * S;
    NS = N * S;

    // Make resource
    mean = ws()->CreateTensor("/mnt/" + anchor() + "/gn/mean");
    var = ws()->CreateTensor("/mnt/" + anchor() + "/gn/var");
    x_norm = ws()->CreateTensor("/mnt/" + anchor() + "/gn/x_norm");

    // Reshape
    mean->Reshape({ NG });
    var->Reshape({ NG });
    nc.Reshape({ NC });
    x_norm->ReshapeLike(Input(0));
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void FusedGroupNormOp<Context>::RunOnDevice() {
    Setup();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}


DEPLOY_CPU(FusedGroupNorm);
#ifdef WITH_CUDA
DEPLOY_CUDA(FusedGroupNorm);
#endif
OPERATOR_SCHEMA(FusedGroupNorm).NumInputs(3).NumOutputs(1);

template <class Context> template <typename T>
void FusedGroupNormGradientOp<Context>::RunWithType() {
    DECLARE_MULTIPLIER(MXmult, std::max(NS, CGS));

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Sdata = Input(1).template data<T, Context>();
    auto* Tmean = mean->template mutable_data<T, Context>();
    auto* Tvar = var->template mutable_data<T, Context>();
    auto* NCdata = nc.template mutable_data<T, Context>();
    auto* XNorm_data = x_norm->template data<T, Context>();
    auto* WSdata = ws()->template caches<T, Context>({ x_norm->count() })[0];

    // Gradient w.r.t. scale
    if (Output(1)->name() != "ignore") {
        auto* dSdata = Output(1)->template mutable_data<T, Context>();
        math::Mul<T, Context>(x_norm->count(),
            XNorm_data, dYdata, WSdata, ctx());
        if (data_format == "NCHW") {
            math::Gemv<T, Context>(
                CblasNoTrans, NC, S,
                    1.f, WSdata, MXmult,
                        0.f, NCdata, ctx());
            math::Gemv<T, Context>(
                CblasTrans, N, C,
                    1.f, NCdata, MXmult,
                        1.f, dSdata, ctx());
        } else if (data_format == "NHWC") {
            math::Gemv<T, Context>(
                CblasTrans, NS, C,
                    1.f, WSdata, MXmult,
                        1.f, dSdata, ctx());
        }
    }

    // Gradient w.r.t. bias
    if (Output(2)->name() != "ignore") {
        auto* dBdata = Output(2)->template mutable_data<T, Context>();
        if (data_format == "NCHW") {
            math::Gemv<T, Context>(
                CblasNoTrans, NC, S,
                    1.f, dYdata, MXmult,
                        0.f, NCdata, ctx());
            math::Gemv<T, Context>(
                CblasTrans, N, C,
                    1.f, NCdata, MXmult,
                        1.f, dBdata, ctx());
        } else if (data_format == "NHWC") {
            math::Gemv<T, Context>(
                CblasTrans, NS, C,
                    1.f, dYdata, MXmult,
                        1.f, dBdata, ctx());
        }
    }

    // Gradient w.r.t. x
    if (Output(0)->name() != "ignore") {
         // scale * dY
         if (data_format == "NCHW") {
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    N, C, 1,
                        1.f, MXmult, Sdata,
                            0.f, NCdata, ctx());
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    NC, S, 1,
                        1.f, NCdata, MXmult,
                            0.f, WSdata, ctx());
         } else if (data_format == "NHWC") {
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    NS, C, 1,
                        1.f, MXmult, Sdata,
                            0.f, WSdata, ctx());
         }
         math::Mul<T, Context>(x_norm->count(),
             WSdata, dYdata, WSdata, ctx());

         // Sum of x_hat * (dl / dx_hat)
         math::Mul<T, Context>(x_norm->count(),
             XNorm_data, WSdata, dXdata, ctx());
         if (data_format == "NCHW") {
             math::Gemv<T, Context>(
                 CblasNoTrans, NG, CGS,
                    1.f, dXdata, MXmult,
                        0.f, Tmean, ctx());
         } else if (data_format == "NHWC") {
             NOT_IMPLEMENTED;
         }

         // x_hat times the sum
         if (data_format == "NCHW") {
             math::Gemm<T, Context>(
                 CblasNoTrans, CblasNoTrans,
                    NG, CGS, 1,
                        1.f, Tmean, MXmult,
                            0.f, dXdata, ctx());
         } else if (data_format == "NHWC") {
             NOT_IMPLEMENTED;
         }
         math::Mul<T, Context>(x_norm->count(),
             XNorm_data, dXdata, dXdata, ctx());

        // Subtract the average of x_hat times the sum
        if (data_format == "NCHW") {
            math::Gemv<T, Context>(
                CblasNoTrans, NG, CGS,
                    1.f, WSdata, MXmult,
                        0.f, Tmean, ctx());
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    NG, CGS, 1,
                        1.f, Tmean, MXmult,
                            1.f, dXdata, ctx());
        } else if (data_format == "NHWC") {
            NOT_IMPLEMENTED;
        }
        math::Axpby<T, Context>(x_norm->count(),
            1.f, WSdata, -1.f / CGS, dXdata, ctx());

        // Multiply with the inverse std
         if (data_format == "NCHW") {
             math::Gemm<T, Context>(
                 CblasNoTrans, CblasNoTrans,
                    NG, CGS, 1,
                        1.f, Tvar, MXmult,
                            0.f, WSdata, ctx());
        } else if (data_format == "NHWC") {
             NOT_IMPLEMENTED;
        }
        // Divide by stddev
        math::Div<T, Context>(Output(0)->count(),
            dXdata, WSdata, dXdata, ctx());
    }
}

template <class Context>
void FusedGroupNormGradientOp<Context>::Setup() {
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
    if (group == C && Input(0).ndim() == 2)    //  InstanceNorm
        LOG(WARNING) << "The 2d input will output all zeros.";
    NC = N * C;
    NG = N * group;
    S = Input(0).count() / NC;
    CGS = (C / group) * S;
    NS = N * S;

    // Make resource
    mean = ws()->GetTensor("/mnt/" + anchor() + "/gn/mean");
    var = ws()->GetTensor("/mnt/" + anchor() + "/gn/var");
    x_norm = ws()->GetTensor("/mnt/" + anchor() + "/gn/x_norm");

    // Reshape
    nc.Reshape({ NC });
    Output(0)->ReshapeLike(Input(0));  // dX
    Output(1)->ReshapeLike(Input(1));  // dScale
    Output(2)->ReshapeLike(Input(1));  // dBias
}

template <class Context>
void FusedGroupNormGradientOp<Context>::RunOnDevice() {
    Setup();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(FusedGroupNormGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(FusedGroupNormGradient);
#endif
OPERATOR_SCHEMA(FusedGroupNormGradient).NumInputs(3).NumOutputs(3);

class GetFusedGroupNormGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetFusedGroupNormGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0), GI(1), GI(2)});
    }
};
REGISTER_GRADIENT(FusedGroupNorm, GetFusedGroupNormGradient);

}  // namespace dragon