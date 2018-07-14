#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/math_functions.h"
#include "operators/norm/batch_renorm_op.h"

namespace dragon {

template <class Context> template <typename T>
void BatchRenormOp<Context>::TrainingRunWithType() {
    TENSOR_FILL(Input(1), vector<TIndex>(1, C));  //  history_mean
    TENSOR_FILL(Input(2), vector<TIndex>(1, C));  //  history_var

    DECLARE_MULTIPLIER(MXmult, NS);

    auto* Hmean = Input(1).template mutable_data<T, Context>();
    auto* Hvar = Input(2).template mutable_data<T, Context>();
    auto* Tmean = mean.template mutable_data<T, Context>();
    auto* Tvar = var->template mutable_data<T, Context>();
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    auto* NCdata = nc.template mutable_data<T, Context>();
    auto* WSdata = ws()->template caches<T, Context>({ Input(0).count() })[0];
    ctx().template Copy<T, Context, Context>(Input(0).count(), Ydata, Xdata);

    auto* Td = d.template mutable_data<T, Context>();
    auto* Tr = r->template mutable_data<T, Context>();
    auto* THmean = t_h_mean.template mutable_data<T, Context>();
    auto* THvar = t_h_var.template mutable_data<T, Context>();

    //  scale the mean and variance if necessary
    if (mode == "CAFFE") {
        CHECK_EQ(InputSize(), 4)
            << "\nThe number of inputs should be 4 if use CAFFE mode.";
        TENSOR_FILL(Input(3), vector<TIndex>(1, 1));
        auto* hFact_data = Input(3).template mutable_data<T, CPUContext>();
        const float factor = dragon_cast<float, T>(hFact_data[0]);
        const float scale = factor == 0 ? 0 : 1.0 / factor;
        math::Scale<T, Context>(mean.count(), scale, Hmean, THmean, &ctx());
        math::Scale<T, Context>(mean.count(), scale, Hvar, THvar, &ctx());
    } else {
       ctx().template Copy<T, Context, Context>(mean.count(), THmean, Hmean);
       ctx().template Copy<T, Context, Context>(var->count(), THvar, Hvar);
    }

    //  compute mean
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(
            CblasNoTrans, NC, S,
                1.0 / NS, Xdata, MXmult,
                    0.0, NCdata, &ctx());
        math::Gemv<T, Context>(
            CblasTrans, N, C,
                1.0, NCdata, MXmult,
                    0.0, Tmean, &ctx());
    } else if (data_format == "NHWC") {
        math::Gemv<T, Context>(
            CblasTrans, NS, C,
                1.0 / NS, Xdata, MXmult,
                    0.0, Tmean, &ctx());
    }

    //  subtract mean
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                N, C, 1,
                    1.0, MXmult, Tmean,
                        0.0, NCdata, &ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NC, S, 1,
                    -1.0, NCdata, MXmult,
                        1.0, Ydata, &ctx());
    } else if (data_format == "NHWC") {
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans,
                NS, C, 1,
                    -1.0, MXmult, Tmean,
                        1.0, Ydata, &ctx());
    }

    //  compute variance
    //  note that we use VAR(X) = E((X - EX) ^ 2)
    math::Square<T, Context>(Output(0)->count(), Ydata, WSdata);
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(
            CblasNoTrans, NC, S,
                1.0 / NS, WSdata, MXmult,
                    0.0, NCdata, &ctx());
        math::Gemv<T, Context>(
            CblasTrans, N, C,
                1.0, NCdata, MXmult,
                    0.0, Tvar, &ctx());
    } else if (data_format == "NHWC") {
        math::Gemv<T, Context>(
            CblasTrans, NS, C,
                1.0 / NS, WSdata, MXmult,
                    0.0, Tvar, &ctx());
    }

    //  compute moving average
    if (!is_recomputing) {
        if (mode == "CAFFE") {
            CHECK_EQ(InputSize(), 4)
                << "\nThe number of inputs should be 4 if use CAFFE mode.";
            TENSOR_FILL(Input(3), vector<TIndex>(1, 1));
            auto* hFact_data = Input(3).template mutable_data<T, CPUContext>();
            float factor = dragon_cast<float, T>(hFact_data[0]);
            factor *= momentum; factor += 1;
            hFact_data[0] = dragon_cast<T, float>(factor);
            int m = Input(0).count() / C;
            float coeff = m > 1 ? float(m) / (m - 1) : 1;
            //  History(X) = Cur(X) + momentum * History(X)
            math::Axpby<T, Context>(mean.count(),
                1.0, Tmean, momentum, Hmean, &ctx());
            math::Axpby<T, Context>(var->count(),
                coeff, Tvar, momentum, Hvar, &ctx());
        } else {
            //  History(X) = (1 - momentum) * Cur(X) + momentum * History(X)
            math::Axpby<T, Context>(mean.count(),
                1.0 - momentum, Tmean, momentum, Hmean, &ctx());
            math::Axpby<T, Context>(var->count(),
                1.0 - momentum, Tvar, momentum, Hvar, &ctx());
        }
    }

    //  compute stddev
    math::AddScalar<T, Context>(var->count(), eps, Tvar);
    math::Sqrt<T, Context>(var->count(), Tvar, Tvar);

     //  divide by stddev
    if (data_format == "NCHW") {
          math::Gemm<T, Context>(
              CblasNoTrans, CblasNoTrans,
                  N, C, 1,
                      1.0, MXmult, Tvar,
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
                    1.0, MXmult, Tvar,
                        0.0, WSdata, &ctx());
    }
    math::Div<T, Context>(Output(0)->count(), Ydata, WSdata, Ydata);

    //  compute renorm
    if (!is_recomputing) {
        //  compute history stddev
        math::AddScalar<T, Context>(var->count(), eps, THvar);
        math::Sqrt<T, Context>(var->count(), THvar, THvar);

        //  compute r
        math::Div<T, Context>(var->count(), Tvar, THvar, Tr);
        math::Clip<T, Context>(var->count(), 1.0 / t_r_max, t_r_max, Tr);

        //  compute d
        math::Sub<T, Context>(mean.count(), Tmean, THmean, Td);
        math::Div<T, Context>(mean.count(), Td, THvar, Td);
        math::Clip<T, Context>(mean.count(), -t_d_max, t_d_max, Td);

        //  update the bound of r & d
        t_r_max = r_max / (1.0 + (r_max - 1.0) * exp(-t_val));
        t_d_max = d_max / (1.0 + (d_max - 1.0) * exp(-2 * t_val));
        t_val += t_delta;
    }

    //  apply renorm
    //  store x_norm for backward
    auto* XNorm_data = x_norm->template mutable_data<T, Context>();
    ctx().template Copy<T, Context, Context>(
        Output(0)->count(), XNorm_data, Ydata);

    //  correction: mul by r
    if (data_format == "NCHW") {
          math::Gemm<T, Context>(
              CblasNoTrans, CblasNoTrans,
                  N, C, 1,
                      1.0, MXmult, Tr,
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
                    1.0, MXmult, Tr,
                        0.0, WSdata, &ctx());
    }
    math::Mul<T, Context>(Output(0)->count(), Ydata, WSdata, Ydata);

    //  correction: add by d
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                N, C, 1,
                    1.0, MXmult, Td,
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
                    1.0, MXmult, Td,
                        1.0, Ydata, &ctx());
    }
}

template <class Context> template <typename T>
void BatchRenormOp<Context>::InferenceRunWithType() {
    TENSOR_FILL(Input(1), vector<TIndex>(1, C));  //  history_mean
    TENSOR_FILL(Input(2), vector<TIndex>(1, C));  //  history_var

    DECLARE_MULTIPLIER(MXmult, NS);

    auto* Hmean = Input(1).template mutable_data<T, Context>();
    auto* Hvar = Input(2).template mutable_data<T, Context>();
    auto* Tmean = mean.template mutable_data<T, Context>();
    auto* Tvar = var->template mutable_data<T, Context>();
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    auto* NCdata = nc.template mutable_data<T, Context>();
    auto* WSdata = ws()->template caches<T, Context>({ Input(0).count() })[0];
    ctx().template Copy<T, Context, Context>(Input(0).count(), Ydata, Xdata);

    //  scale the mean and variance if necessary
    if (mode == "CAFFE") {
        CHECK_EQ(InputSize(), 4)
            << "\nThe number of inputs should be 4 if use CAFFE mode.";
        TENSOR_FILL(Input(3), vector<TIndex>(1, 1));
        auto* hFact_data = Input(3).template mutable_data<T, CPUContext>();
        const float factor = dragon_cast<float, T>(hFact_data[0]);
        const float scale = factor == 0 ? 0 : 1.0 / factor;
        math::Scale<T, Context>(mean.count(), scale, Hmean, Tmean, &ctx());
        math::Scale<T, Context>(var->count(), scale, Hvar, Tvar, &ctx());
    } else {
       ctx().template Copy<T, Context, Context>(mean.count(), Tmean, Hmean);
       ctx().template Copy<T, Context, Context>(var->count(), Tvar, Hvar);
    }

    //  subtract mean
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                N, C, 1,
                    1.0, MXmult, Tmean,
                        0.0, NCdata, &ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans, NC, S, 1,
                -1.0, NCdata, MXmult,
                    1.0, Ydata, &ctx());
    } else if (data_format == "NHWC") {
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans,
                NS, C, 1,
                    -1.0, MXmult, Tmean,
                        1.0, Ydata, &ctx());
    }

    //  compute stddev
    math::AddScalar<T, Context>(var->count(), eps, Tvar);
    math::Sqrt<T, Context>(var->count(), Tvar, Tvar);

    //  divide by stddev
    if (data_format == "NCHW") {
          math::Gemm<T, Context>(
              CblasNoTrans, CblasNoTrans,
                  N, C, 1,
                      1.0, MXmult, Tvar,
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
                    1.0, MXmult, Tvar,
                        0.0, WSdata, &ctx());
    }
    math::Div<T, Context>(Output(0)->count(), Ydata, WSdata, Ydata);
}

template <class Context>
void BatchRenormOp<Context>::Setup() {
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
    var = ws()->CreateTensor("/mnt/" + anchor() + "/bn/var");
    r = ws()->CreateTensor("/mnt/" + anchor() + "/bn/r");
    x_norm = ws()->CreateTensor("/mnt/" + anchor() + "/bn/x_norm");

    //  reshape
    mean.Reshape({ C });
    var->Reshape({ C });
    d.Reshape({ C });
    r->Reshape({ C });
    t_h_mean.Reshape({ C });
    t_h_var.Reshape({ C });
    nc.Reshape({ NC });
    x_norm->ReshapeLike(Input(0));
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void BatchRenormOp<Context>::RunOnDevice() {
    Setup();

    if (XIsType(Input(0), float)) {
        if (use_global_stats) InferenceRunWithType<float>();
        else TrainingRunWithType<float>();
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(BatchRenorm);
#ifdef WITH_CUDA
DEPLOY_CUDA(BatchRenorm);
#endif
OPERATOR_SCHEMA(BatchRenorm).NumInputs(3, 4).NumOutputs(1); 

template <class Context> template <typename T>
void BatchRenormGradientOp<Context>::TrainingRunWithType() {
    DECLARE_MULTIPLIER(MXmult, NS);

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Tmean = mean.template mutable_data<T, Context>();
    auto* Tvar = var->template mutable_data<T, Context>();
    auto* NCdata = nc.template mutable_data<T, Context>();
    auto* Tr = r->template data<T, Context>();
    auto* XNorm_data = x_norm->template data<T, Context>();
    auto* WSdata = ws()->template caches<T, Context>({ x_norm->count() })[0];

    //  buffer <- dE/dY \cdot r
    if (data_format == "NCHW") {
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans,
                 N, C, 1,
                     1.0, MXmult, Tr,
                         0.0, NCdata, &ctx());
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans, 
                 NC, S, 1,
                     1.0, NCdata, MXmult,
                         0.0, WSdata, &ctx());
    } else if (data_format == "NWHC") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NS, C, 1,
                    1.0, MXmult, Tr,
                        0.0, WSdata, &ctx());
    }
    math::Mul<T, Context>(Output(0)->count(), dYdata, WSdata, WSdata);

    //  sum(dE/dY \cdot Y)
    math::Mul<T, Context>(Output(0)->count(), XNorm_data, WSdata, dXdata);
    if (data_format == "NCHW") {
         math::Gemv<T, Context>(
             CblasNoTrans, NC, S,
                 1.0, dXdata, MXmult,
                     0.0, NCdata, &ctx());
         math::Gemv<T, Context>(
             CblasTrans, N, C,
                 1.0, NCdata, MXmult,
                     0.0, Tmean, &ctx());
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans,
                 N, C, 1,
                     1.0, MXmult, Tmean,
                         0.0, NCdata, &ctx());
         math::Gemm<T, Context>(
             CblasNoTrans, CblasNoTrans,
                 NC, S, 1,
                     1.0, NCdata, MXmult,
                         0.0, dXdata, &ctx());
    } else if (data_format == "NHWC") {
        math::Gemv<T, Context>(
            CblasTrans, NS, C,
                1.0, dXdata, MXmult,
                    0.0, Tmean, &ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NS, C, 1,
                    1.0, MXmult, Tmean,
                        0.0, dXdata, &ctx());
    }

    //  sum(dE/dY \cdot Y) \cdot Y  
    math::Mul<T, Context>(Output(0)->count(), XNorm_data, dXdata, dXdata);

    //  sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(
            CblasNoTrans, NC, S,
                1.0, WSdata, MXmult,
                    0.0, NCdata, &ctx());
        math::Gemv<T, Context>(
            CblasTrans, N, C,
                1.0, NCdata, MXmult,
                    0.0, Tmean, &ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                N, C, 1,
                    1.0, MXmult, Tmean,
                        0.0, NCdata, &ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NC, S, 1,
                    1.0, NCdata, MXmult,
                        1.0, dXdata, &ctx());
    } else if (data_format == "NHWC") {
        math::Gemv<T, Context>(
            CblasTrans, NS, C,
                1.0, WSdata, MXmult,
                    0.0, Tmean, &ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NS, C, 1,
                    1.0, MXmult, Tmean,
                        1.0, dXdata, &ctx());
    }

    //  dE/dY - mean(dE/dY)- mean(dE/dY \cdot Y) \cdot Y
    //  = dE/dY - mean(sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y)
    math::Axpby<T, Context>(Output(0)->count(),
        1.0, WSdata, -1.0 / NS, dXdata, &ctx());

    //  divide by stddev
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                N, C, 1,
                    1.0, MXmult, Tvar,
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
                    1.0, MXmult, Tvar,
                        0.0, WSdata, &ctx());
    }

    math::Div<T, Context>(Output(0)->count(), dXdata, WSdata, dXdata);
    x_norm->Reset();
}

template <class Context> template <typename T>
void BatchRenormGradientOp<Context>::InferenceRunWithType() {
    DECLARE_MULTIPLIER(MXmult, NS);

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Tvar = var->template mutable_data<T, Context>();
    auto* NCdata = nc.template mutable_data<T, Context>();
    auto* WSdata = ws()->template caches<T, Context>({ Output(0)->count() })[0];

    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                N, C, 1,
                    1.0, MXmult, Tvar,
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
                    1.0, MXmult, Tvar,
                        0.0, WSdata, &ctx());
    }

    math::Div<T, Context>(Output(0)->count(), dYdata, WSdata, dXdata);
}

template <class Context>
void BatchRenormGradientOp<Context>::Setup() {
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
    var = ws()->GetTensor("/mnt/" + anchor() + "/bn/var");
    r = ws()->GetTensor("/mnt/" + anchor() + "/bn/r");
    x_norm = ws()->GetTensor("/mnt/" + anchor() + "/bn/x_norm");

    //  reshape
    mean.ReshapeLike(*var);
    nc.Reshape({ NC });
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void BatchRenormGradientOp<Context>::RunOnDevice() {
    Setup();

    if (XIsType(Input(0), float)) {
        if (use_global_stats) InferenceRunWithType<float>();
        else TrainingRunWithType<float>();
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(BatchRenormGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(BatchRenormGradient);
#endif
OPERATOR_SCHEMA(BatchRenormGradient).NumInputs(3).NumOutputs(1);

class GetBatchRenormGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetBatchRenormGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), O(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(BatchRenorm, GetBatchRenormGradient);

}    // namespace dragon