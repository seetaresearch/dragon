#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/math_functions.h"
#include "operators/norm/batch_norm_op.h"

namespace dragon {

template <class Context> template <typename T>
void FusedBatchNormOp<Context>::TrainingRunWithType() {
    TENSOR_FILL(Input(1), vector<TIndex>(1, C));  //  history_mean
    TENSOR_FILL(Input(2), vector<TIndex>(1, C));  //  history_var
    TENSOR_FILL(Input(3), vector<TIndex>(1, C));  //  scale
    TENSOR_FILL(Input(4), vector<TIndex>(1, C));  //  bias

    DECLARE_MULTIPLIER(MXmult, NS);

    auto* Hmean = Input(1).template mutable_data<T, Context>();
    auto* Hvar = Input(2).template mutable_data<T, Context>();
    auto* Sdata = Input(3).template data<T, Context>();
    auto* Bdata = Input(4).template data<T, Context>();
    auto* Tmean = mean->template mutable_data<T, Context>();
    auto* Tvar = var->template mutable_data<T, Context>();
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    auto* NCdata = nc.template mutable_data<T, Context>();
    auto* WSdata = ws()->template caches<T, Context>({ Input(0).count() })[0];
    ctx()->template Copy<T, Context, Context>(Output(0)->count(), Ydata, Xdata);

    //  compute mean
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(
            CblasNoTrans, NC, S,
                1.0 / NS, Xdata, MXmult,
                    0.0, NCdata, ctx());
        math::Gemv<T, Context>(
            CblasTrans, N, C,
                1.0, NCdata, MXmult,
                    0.0, Tmean, ctx());
    } else if (data_format == "NHWC") {
        math::Gemv<T, Context>(
            CblasTrans, NS, C,
                1.0 / NS, Xdata, MXmult,
                    0.0, Tmean, ctx());
    }

    //  subtract mean
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                N, C, 1,
                    1.0, MXmult, Tmean,
                        0.0, NCdata, ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NC, S, 1,
                    -1.0, NCdata, MXmult,
                        1.0, Ydata, ctx());
    } else if (data_format == "NHWC") {
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans,
                NS, C, 1,
                    -1.0, MXmult, Tmean,
                        1.0, Ydata, ctx());
    }

    //  compute variance
    //  note that we use VAR(X) = E((X - EX) ^ 2)
    math::Square<T, Context>(Output(0)->count(), Ydata, WSdata, ctx());
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(
            CblasNoTrans, NC, S,
                1.0 / NS, WSdata, MXmult,
                    0.0, NCdata, ctx());
        math::Gemv<T, Context>(
            CblasTrans, N, C,
                1.0, NCdata, MXmult,
                    0.0, Tvar, ctx());
    } else if (data_format == "NHWC") {
        math::Gemv<T, Context>(
            CblasTrans, NS, C,
                1.0 / NS, WSdata, MXmult,
                    0.0, Tvar, ctx());
    }

    //  compute moving average
    if (!is_recomputing) {
        //  History(X) = (1 - momentum) * Cur(X) + momentum * History(X)
        math::Axpby<T, Context>(mean->count(),
            1.0 - momentum, Tmean, momentum, Hmean, ctx());
        math::Axpby<T, Context>(var->count(),
            1.0 - momentum, Tvar, momentum, Hvar, ctx());
    }

    //  compute stddev
    math::AddScalar<T, Context>(var->count(), eps, Tvar, ctx());
    math::Sqrt<T, Context>(var->count(), Tvar, Tvar, ctx());

    //  divide by stddev
    if (data_format == "NCHW") {
          math::Gemm<T, Context>(
              CblasNoTrans, CblasNoTrans,
                  N, C, 1,
                      1.0, MXmult, Tvar,
                          0.0, NCdata, ctx());
          math::Gemm<T, Context>(
              CblasNoTrans, CblasNoTrans,
                  NC, S, 1,
                      1.0, NCdata, MXmult,
                          0.0, WSdata, ctx());
    } else if (data_format == "NHWC") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NS, C, 1,
                    1.0, MXmult, Tvar,
                        0.0, WSdata, ctx());
    }
    math::Div<T, Context>(Output(0)->count(),
        Ydata, WSdata, Ydata, ctx());

    //  store x_norm for backward
    auto* XNorm_data = x_norm->template mutable_data<T, Context>();
    ctx()->template Copy<T, Context, Context>(
        Output(0)->count(), XNorm_data, Ydata);

    // scale
    if (data_format == "NCHW") {
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans,
                 N, C, 1,
                     1.0, MXmult, Sdata,
                         0.0, NCdata, ctx());
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans,
                NC, S, 1,
                    1.0, NCdata, MXmult,
                        0.0, WSdata, ctx());
    } else if (data_format == "NHWC") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NS, C, 1,
                    1.0, MXmult, Sdata,
                        0.0, WSdata, ctx());
    }
    math::Mul<T, Context>(Output(0)->count(),
        Ydata, WSdata, Ydata, ctx());

    // shift
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                N, C, 1,
                    1.0, MXmult, Bdata,
                        0.0, NCdata, ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NC, S, 1,
                    1.0, NCdata, MXmult,
                        1.0, Ydata, ctx());
    } else if (data_format == "NHWC") {
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans,
                 NS, C, 1,
                     1.0, MXmult, Bdata,
                         1.0, Ydata, ctx());
    }
}

template <class Context> template <typename T>
void FusedBatchNormOp<Context>::InferenceRunWithType() {
    TENSOR_FILL(Input(1), vector<TIndex>(1, C));  //  history_mean
    TENSOR_FILL(Input(2), vector<TIndex>(1, C));  //  history_var
    TENSOR_FILL(Input(3), vector<TIndex>(1, C));  //  scale
    TENSOR_FILL(Input(4), vector<TIndex>(1, C));  //  bias

    DECLARE_MULTIPLIER(MXmult, NS);

    auto* Hmean = Input(1).template mutable_data<T, Context>();
    auto* Hvar = Input(2).template mutable_data<T, Context>();
    auto* Sdata = Input(3).template data<T, Context>();
    auto* Bdata = Input(4).template data<T, Context>();
    auto* Tmean = mean->template mutable_data<T, Context>();
    auto* Tvar = var->template mutable_data<T, Context>();
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    auto* NCdata = nc.template mutable_data<T, Context>();
    auto* WSdata = ws()->template caches<T, Context>({ Input(0).count() })[0];
    ctx()->template Copy<T, Context, Context>(Input(0).count(), Ydata, Xdata);
    ctx()->template Copy<T, Context, Context>(mean->count(), Tmean, Hmean);
    ctx()->template Copy<T, Context, Context>(var->count(), Tvar, Hvar);

    //  subtract mean
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                N, C, 1,
                    1.0, MXmult, Tmean,
                        0.0, NCdata, ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NC, S, 1,
                    -1.0, NCdata, MXmult,
                        1.0, Ydata, ctx());
    } else if (data_format == "NHWC") {
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans,
                NS, C, 1,
                    -1.0, MXmult, Tmean,
                        1.0, Ydata, ctx());
    }

    //  compute stddev
    math::AddScalar<T, Context>(var->count(), eps, Tvar, ctx());
    math::Sqrt<T, Context>(var->count(), Tvar, Tvar, ctx());

    //  divide by stddev
    if (data_format == "NCHW") {
          math::Gemm<T, Context>(
              CblasNoTrans, CblasNoTrans,
                  N, C, 1,
                      1.0, MXmult, Tvar,
                          0.0, NCdata, ctx());
          math::Gemm<T, Context>(
              CblasNoTrans, CblasNoTrans,
                  NC, S, 1,
                      1.0, NCdata, MXmult,
                          0.0, WSdata, ctx());
    } else if (data_format == "NHWC") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NS, C, 1,
                    1.0, MXmult, Tvar,
                        0.0, WSdata, ctx());
    }
    math::Div<T, Context>(Output(0)->count(),
        Ydata, WSdata, Ydata, ctx());

    // scale
    if (data_format == "NCHW") {
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans,
                N, C, 1,
                    1.0, MXmult, Sdata,
                        0.0, NCdata, ctx());
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans,
                NC, S, 1,
                    1.0, NCdata, MXmult,
                        0.0, WSdata, ctx());
    } else if (data_format == "NHWC") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NS, C, 1,
                    1.0, MXmult, Sdata,
                        0.0, WSdata, ctx());
    }
    math::Mul<T, Context>(Output(0)->count(),
        Ydata, WSdata, Ydata, ctx());

    // shift
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                N, C, 1,
                    1.0, MXmult, Bdata,
                        0.0, NCdata, ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NC, S, 1,
                    1.0, NCdata, MXmult,
                        1.0, Ydata, ctx());
    } else if (data_format == "NHWC") {
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans,
                 NS, C, 1,
                     1.0, MXmult, Bdata,
                         1.0, Ydata, ctx());
    }
}

template <class Context>
void FusedBatchNormOp<Context>::Setup() {
    //  determine the mode
    if (use_stats == -1) use_global_stats = phase() == "TEST" ? true : false;
    else use_global_stats = use_stats == 1 ? true : false;
    is_recomputing = ws()->GetTensor("/opt/mirror_stage/recompute_flag")
                         ->template data<bool, CPUContext>()[0];

    //  determine the data format
    TIndex channel_axis = axis;
    data_format = "NCHW";
    if (channel_axis == -1) channel_axis += (int)Input(0).ndim();
    if (channel_axis + 1 == (int)Input(0).ndim()) data_format = "NHWC";
    N = Input(0).dim(0);
    C = Input(0).dim(channel_axis);
    NC = N * C;
    S = Input(0).count() / NC;
    NS = N * S;

    //  make resource
    mean = ws()->CreateTensor("/mnt/" + anchor() + "/bn/mean");
    var = ws()->CreateTensor("/mnt/" + anchor() + "/bn/var");
    x_norm = ws()->CreateTensor("/mnt/" + anchor() + "/bn/x_norm");

    //  reshape
    mean->Reshape({ C });
    var->Reshape({ C });
    nc.Reshape({ NC });
    x_norm->ReshapeLike(Input(0));
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void FusedBatchNormOp<Context>::RunOnDevice() {
    Setup();

    if (XIsType(Input(0), float)) {
        if (use_global_stats) InferenceRunWithType<float>();
        else TrainingRunWithType<float>();
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}


DEPLOY_CPU(FusedBatchNorm);
#ifdef WITH_CUDA
DEPLOY_CUDA(FusedBatchNorm);
#endif
OPERATOR_SCHEMA(FusedBatchNorm).NumInputs(5).NumOutputs(1);

template <class Context> template <typename T>
void FusedBatchNormGradientOp<Context>::TrainingRunWithType() {
    DECLARE_MULTIPLIER(MXmult, NS);

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Sdata = Input(3).template data<T, Context>();
    auto* Tmean = mean->template mutable_data<T, Context>();
    auto* Tvar = var->template mutable_data<T, Context>();
    auto* NCdata = nc.template mutable_data<T, Context>();
    auto* XNorm_data = x_norm->template data<T, Context>();
    auto* WSdata = ws()->template caches<T, Context>({ x_norm->count() })[0];

    // gradient w.r.t. scale
    if (Output(1)->name() != "ignore") {
        auto* dSdata = Output(1)->template mutable_data<T, Context>();
        math::Mul<T, Context>(x_norm->count(),
            XNorm_data, dYdata, WSdata, ctx());
        if (data_format == "NCHW") {
            math::Gemv<T, Context>(
                CblasNoTrans, NC, S,
                    1.0, WSdata, MXmult,
                        0.0, NCdata, ctx());
            math::Gemv<T, Context>(
                CblasTrans, N, C,
                    1.0, NCdata, MXmult,
                        1.0, dSdata, ctx());
        } else if (data_format == "NHWC") {
            math::Gemv<T, Context>(
                CblasTrans, NS, C,
                    1.0, WSdata, MXmult,
                        1.0, dSdata, ctx());
        }
    }

    // gradient w.r.t. bias
    if (Output(2)->name() != "ignore") {
        auto* dBdata = Output(2)->template mutable_data<T, Context>();
        if (data_format == "NCHW") {
            math::Gemv<T, Context>(
                CblasNoTrans, NC, S,
                    1.0, dYdata, MXmult,
                        0.0, NCdata, ctx());
            math::Gemv<T, Context>(
                CblasTrans, N, C,
                    1.0, NCdata, MXmult,
                        1.0, dBdata, ctx());
        } else if (data_format == "NHWC") {
            math::Gemv<T, Context>(
                CblasTrans, NS, C,
                    1.0, dYdata, MXmult,
                        1.0, dBdata, ctx());
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
                            0.0, NCdata, ctx());
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    NC, S, 1,
                        1.0, NCdata, MXmult,
                            0.0, WSdata, ctx());
         } else if (data_format == "NHWC") {
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    NS, C, 1,
                        1.0, MXmult, Sdata,
                            0.0, WSdata, ctx());
         }
         math::Mul<T, Context>(x_norm->count(),
             WSdata, dYdata, WSdata, ctx());

         // sum of x_hat * (dl / dx_hat)
         math::Mul<T, Context>(x_norm->count(),
             XNorm_data, WSdata, dXdata, ctx());
         if (data_format == "NCHW") {
             math::Gemv<T, Context>(
                 CblasNoTrans, NC, S,
                     1.0, dXdata, MXmult,
                         0.0, NCdata, ctx());
             math::Gemv<T, Context>(
                 CblasTrans, N, C,
                    1.0, NCdata, MXmult,
                         0.0, Tmean, ctx());
         } else if (data_format == "NHWC") {
             math::Gemv<T, Context>(
                 CblasTrans, NS, C,
                     1.0, dXdata, MXmult,
                         0.0, Tmean, ctx());
         }

         // x_hat times the sum
         if (data_format == "NCHW") {
             math::Gemm<T, Context>(
                 CblasNoTrans, CblasNoTrans,
                     N, C, 1,
                         1.0, MXmult, Tmean,
                             0.0, NCdata, ctx());
             math::Gemm<T, Context>(
                 CblasNoTrans, CblasNoTrans,
                     NC, S, 1,
                         1.0, NCdata, MXmult,
                             0.0, dXdata, ctx());
         } else if (data_format == "NHWC") {
             math::Gemm<T, Context>(
                 CblasNoTrans, CblasNoTrans,
                     NS, C, 1,
                         1.0, MXmult, Tmean,
                             0.0, dXdata, ctx());
         }
         math::Mul<T, Context>(x_norm->count(),
             XNorm_data, dXdata, dXdata, ctx());

        // subtract the average of x_hat times the sum
        if (data_format == "NCHW") {
            math::Gemv<T, Context>(
                CblasNoTrans, NC, S,
                    1.0, WSdata, MXmult,
                        0.0, NCdata, ctx());
            math::Gemv<T, Context>(
                CblasTrans, N, C,
                    1.0, NCdata, MXmult,
                        0.0, Tmean, ctx());
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    N, C, 1,
                        1.0, MXmult, Tmean,
                            0.0, NCdata, ctx());
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    NC, S, 1,
                        1.0, NCdata, MXmult,
                            1.0, dXdata, ctx());
        } else if (data_format == "NHWC") {
            math::Gemv<T, Context>(
                CblasTrans, NS, C,
                    1.0, WSdata, MXmult,
                        0.0, Tmean, ctx());
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    NS, C, 1,
                        1.0, MXmult, Tmean,
                            1.0, dXdata, ctx());
        }
        math::Axpby<T, Context>(x_norm->count(),
            1.0, WSdata, -1.0 / NS, dXdata, ctx());

        // multiply with the inverse std
         if (data_format == "NCHW") {
             math::Gemm<T, Context>(
                 CblasNoTrans, CblasNoTrans,
                     N, C, 1,
                         1.0, MXmult, Tvar,
                             0.0, NCdata, ctx());
             math::Gemm<T, Context>(
                 CblasNoTrans, CblasNoTrans,
                     NC, S, 1,
                         1.0, NCdata, MXmult,
                             0.0, WSdata, ctx());
        } else if (data_format == "NHWC") {
             math::Gemm<T, Context>(
                 CblasNoTrans, CblasNoTrans,
                    NS, C, 1,
                        1.0, MXmult, Tvar,
                            0.0, WSdata, ctx());
        }
        //  divide by stddev
        math::Div<T, Context>(x_norm->count(),
            dXdata, WSdata, dXdata, ctx());
    }
}

template <class Context> template <typename T>
void FusedBatchNormGradientOp<Context>::InferenceRunWithType() {
    DECLARE_MULTIPLIER(MXmult, NS);

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* Sdata = Input(3).template data<T, Context>();
    auto* Tvar = var->template mutable_data<T, Context>();
    auto* NCdata = nc.template mutable_data<T, Context>();

    //  gradient w.r.t. scale
    if (Output(1)->name() != "ignore") 
        LOG(FATAL) << "The gamma should be fixed if using global stats.";
       
    //  gradient w.r.t. bias
    if (Output(2)->name() != "ignore") {
        auto* dBdata = Output(2)->template mutable_data<T, Context>();
        if (data_format == "NCHW") {
            math::Gemv<T, Context>(
                CblasNoTrans, NC, S,
                    1.0, dYdata, MXmult,
                        0.0, NCdata, ctx());
            math::Gemv<T, Context>(
                CblasTrans, N, C,
                    1.0, NCdata, MXmult,
                        1.0, dBdata, ctx());
        } else if (data_format == "NHWC") {
            math::Gemv<T, Context>(
                CblasTrans, NS, C,
                    1.0, dYdata, MXmult,
                        1.0, dBdata, ctx());
            }
    }

    //  gradient w.r.t. x
    if (Output(0)->name() != "ignore") {
        auto* dXdata = Output(0)->template mutable_data<T, Context>();
        auto* WSdata = ws()->template caches<T, Context>({ Input(0).count() })[0];

        //  divide scale by stddev
        math::Div<T, Context>(var->count(), Sdata, Tvar, Tvar, ctx());

        //  compute dE/dY \cot (scale / std(X))
        if (data_format == "NCHW") {
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    N, C, 1,
                        1.0, MXmult, Tvar,
                            0.0, NCdata, ctx());
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    NC, S, 1,
                        1.0, NCdata, MXmult,
                            0.0, WSdata, ctx());
        } else if (data_format == "NHWC") {
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    NS, C, 1,
                        1.0, MXmult, Tvar,
                            0.0, WSdata, ctx());
        }
        math::Mul<T, Context>(Output(0)->count(),
            dYdata, WSdata, dXdata, ctx());
    }
}

template <class Context>
void FusedBatchNormGradientOp<Context>::Setup() {
    //  determine the mode
    if (use_stats == -1) use_global_stats = phase() == "TEST" ? true : false;
    else use_global_stats = use_stats == 1 ? true : false;

    //  determine the data format
    TIndex channel_axis = axis;
    data_format = "NCHW";
    if (channel_axis == -1) channel_axis += (int)Input(0).ndim();
    if (channel_axis + 1 == (int)Input(0).ndim()) data_format = "NHWC";
    N = Input(0).dim(0);
    C = Input(0).dim(channel_axis);
    NC = N * C;
    S = Input(0).count() / NC;
    NS = N * S;

    //  make resource
    mean = ws()->GetTensor("/mnt/" + anchor() + "/bn/mean");
    var = ws()->GetTensor("/mnt/" + anchor() + "/bn/var");
    x_norm = ws()->GetTensor("/mnt/" + anchor() + "/bn/x_norm");

    //  reshape
    nc.Reshape({ NC });
    Output(0)->ReshapeLike(Input(0));  // dX
    Output(1)->ReshapeLike(Input(3));  // dScale
    Output(2)->ReshapeLike(Input(3));  // dBias
}

template <class Context>
void FusedBatchNormGradientOp<Context>::RunOnDevice() {
    Setup();

    if (XIsType(Input(0), float)) {
        if (use_global_stats) InferenceRunWithType<float>();
        else TrainingRunWithType<float>();
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(FusedBatchNormGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(FusedBatchNormGradient);
#endif
OPERATOR_SCHEMA(FusedBatchNormGradient).NumInputs(5).NumOutputs(3);

class GetFusedBatchNormGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetFusedBatchNormGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), I(2), I(3), GO(0)},
            vector<string> {GI(0), GI(3), GI(4)});
    }
};
REGISTER_GRADIENT(FusedBatchNorm, GetFusedBatchNormGradient);

}    // namespace dragon