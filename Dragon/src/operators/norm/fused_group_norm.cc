#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/math_functions.h"
#include "operators/norm/group_norm_op.h"

namespace dragon {

template <class Context> template <typename T>
void FusedGroupNormOp<Context>::RunWithType() {
    TENSOR_FILL(Input(1), vector<TIndex>(1, C));  //  scale
    TENSOR_FILL(Input(2), vector<TIndex>(1, C));  //  bias

    DECLARE_MULTIPLIER(MXmult, std::max(NS, CGS));

    auto* Sdata = Input(1).template data<T, Context>();
    auto* Bdata = Input(2).template data<T, Context>();
    auto* Tmean = mean->template mutable_data<T, Context>();
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
                    0.0, Tmean, &ctx());
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

    //  store x_norm for backward
    auto* XNorm_data = x_norm->template mutable_data<T, Context>();
    ctx().template Copy<T, Context, Context>(
        Output(0)->count(), XNorm_data, Ydata);

    // scale
    if (data_format == "NCHW") {
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans,
                 N, C, 1,
                     1.0, MXmult, Sdata,
                         0.0, NCdata, &ctx());
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans,
                 NC, S, 1,
                     1.0, NCdata, MXmult,
                         0.0, WSdata, &ctx());
    } else if (data_format == "NHWC") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                 NS, C, 1,
                     1.0, MXmult, Sdata,
                         0.0, WSdata, &ctx());
    }
    math::Mul<T, Context>(Output(0)->count(), Ydata, WSdata, Ydata);

    // shift
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                N, C, 1,
                    1.0, MXmult, Bdata,
                        0.0, NCdata, &ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NC, S, 1,
                    1.0, NCdata, MXmult,
                        1.0, Ydata, &ctx());
    } else if (data_format == "NHWC") {
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans,
                NS, C, 1,
                    1.0, MXmult,  Bdata,
                        1.0, Ydata, &ctx());
    }
}

template <class Context>
void FusedGroupNormOp<Context>::Setup() {
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
    if (group == C && Input(0).ndim() == 2)    //  InstanceNorm
        LOG(WARNING) << "The 2d input will output all zeros.";
    NC = N * C;
    NG = N * group;
    S = Input(0).count() / NC;
    CGS = (C / group) * S;
    NS = N * S;

    //  make resource
    mean = ws()->CreateTensor("/mnt/" + anchor() + "/gn/mean");
    var = ws()->CreateTensor("/mnt/" + anchor() + "/gn/var");
    x_norm = ws()->CreateTensor("/mnt/" + anchor() + "/gn/x_norm");

    //  reshape
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
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
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

    // gradient w.r.t. scale
    if (Output(1)->name() != "ignore") {
        auto* dSdata = Output(1)->template mutable_data<T, Context>();
        math::Mul<T, Context>(x_norm->count(), XNorm_data, dYdata, WSdata);
        if (data_format == "NCHW") {
            math::Gemv<T, Context>(
                CblasNoTrans, NC, S,
                    1.0, WSdata, MXmult,
                        0.0, NCdata, &ctx());
            math::Gemv<T, Context>(
                CblasTrans, N, C,
                    1.0, NCdata, MXmult,
                        1.0, dSdata, &ctx());
        } else if (data_format == "NHWC") {
            math::Gemv<T, Context>(
                CblasTrans, NS, C,
                    1.0, WSdata, MXmult,
                        1.0, dSdata, &ctx());
        }
    }

    // gradient w.r.t. bias
    if (Output(2)->name() != "ignore") {
        auto* dBdata = Output(2)->template mutable_data<T, Context>();
        if (data_format == "NCHW") {
            math::Gemv<T, Context>(
                CblasNoTrans, NC, S,
                    1.0, dYdata, MXmult,
                        0.0, NCdata, &ctx());
            math::Gemv<T, Context>(
                CblasTrans, N, C,
                    1.0, NCdata, MXmult,
                        1.0, dBdata, &ctx());
        } else if (data_format == "NHWC") {
            math::Gemv<T, Context>(
                CblasTrans, NS, C,
                    1.0, dYdata, MXmult,
                        1.0, dBdata, &ctx());
        }
    }

    // gradient w.r.t. x
    if (Output(0)->name() != "ignore") {
         // scale * dY
         if (data_format == "NCHW") {
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    N, C, 1,
                        1.0, MXmult, Sdata,
                            0.0, NCdata, &ctx());
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    NC, S, 1,
                        1.0, NCdata, MXmult,
                            0.0, WSdata, &ctx());
         } else if (data_format == "NHWC") {
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    NS, C, 1,
                        1.0, MXmult, Sdata,
                            0.0, WSdata, &ctx());
         }
         math::Mul<T, Context>(x_norm->count(), WSdata, dYdata, WSdata);

         // sum of x_hat * (dl / dx_hat)
         math::Mul<T, Context>(x_norm->count(), XNorm_data, WSdata, dXdata);
         if (data_format == "NCHW") {
             math::Gemv<T, Context>(
                 CblasNoTrans, NG, CGS,
                    1.0, dXdata, MXmult,
                        0.0, Tmean, &ctx());
         } else if (data_format == "NHWC") {
             NOT_IMPLEMENTED;
         }

         // x_hat times the sum
         if (data_format == "NCHW") {
             math::Gemm<T, Context>(
                 CblasNoTrans, CblasNoTrans,
                    NG, CGS, 1,
                        1.0, Tmean, MXmult,
                            0.0, dXdata, &ctx());
         } else if (data_format == "NHWC") {
             NOT_IMPLEMENTED;
         }
         math::Mul<T, Context>(x_norm->count(), XNorm_data, dXdata, dXdata);

        // subtract the average of x_hat times the sum
        if (data_format == "NCHW") {
            math::Gemv<T, Context>(
                CblasNoTrans, NG, CGS,
                    1.0, WSdata, MXmult,
                        0.0, Tmean, &ctx());
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    NG, CGS, 1,
                        1.0, Tmean, MXmult,
                            1.0, dXdata, &ctx());
        } else if (data_format == "NHWC") {
            NOT_IMPLEMENTED;
        }
        math::Axpby<T, Context>(x_norm->count(),
            1.0, WSdata, -1.0 / CGS, dXdata, &ctx());

        // multiply with the inverse std
         if (data_format == "NCHW") {
             math::Gemm<T, Context>(
                 CblasNoTrans, CblasNoTrans,
                    NG, CGS, 1,
                        1.0, Tvar, MXmult,
                            0.0, WSdata, &ctx());
        } else if (data_format == "NHWC") {
             NOT_IMPLEMENTED;
        }
        //  divide by stddev
        math::Div<T, Context>(Output(0)->count(), dXdata, WSdata, dXdata);
    }
}

template <class Context>
void FusedGroupNormGradientOp<Context>::Setup() {
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
    if (group == C && Input(0).ndim() == 2)    //  InstanceNorm
        LOG(WARNING) << "The 2d input will output all zeros.";
    NC = N * C;
    NG = N * group;
    S = Input(0).count() / NC;
    CGS = (C / group) * S;
    NS = N * S;

    //  make resource
    mean = ws()->GetTensor("/mnt/" + anchor() + "/gn/mean");
    var = ws()->GetTensor("/mnt/" + anchor() + "/gn/var");
    x_norm = ws()->GetTensor("/mnt/" + anchor() + "/gn/x_norm");

    //  reshape
    nc.Reshape({ NC });
    Output(0)->ReshapeLike(Input(0));  // dX
    Output(1)->ReshapeLike(Input(1));  // dScale
    Output(2)->ReshapeLike(Input(1));  // dBias
}

template <class Context>
void FusedGroupNormGradientOp<Context>::RunOnDevice() {
    Setup();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
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

}    // namespace dragon