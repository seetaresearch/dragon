#include "operators/norm/group_norm_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/filler.h"

namespace dragon {

template <class Context> template <typename T>
void FusedGroupNormOp<Context>::TrainingRunWithType() {
    INIT_MULTIPLIER(multiplier, NS);
    INIT_MULTIPLIER(num_multiplier, N);
    INIT_MULTIPLIER(spatial_multiplier, S);
    INIT_MULTIPLIER(cgs_multiplier, CGS);
    TENSOR_FILL(Input(1), vector<TIndex>(1, NG));  //  history_mean
    TENSOR_FILL(Input(2), vector<TIndex>(1, NG));  //  history_var
    TENSOR_FILL(Input(3), vector<TIndex>(1, C));  //  scale
    TENSOR_FILL(Input(4), vector<TIndex>(1, C));  //  bias

    auto* hMean_data = Input(1).template mutable_data<T, Context>();
    auto* hVar_data = Input(2).template mutable_data<T, Context>();
    auto* Sdata = Input(3).template data<T, Context>();
    auto* Bdata = Input(4).template data<T, Context>();
    auto* tMean_data = mean->template mutable_data<T, Context>();
    auto* tVar_data = var->template mutable_data<T, Context>();
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    auto* NMul_data = num_multiplier->template data<T, Context>();
    auto* SMul_data = spatial_multiplier->template data<T, Context>();
    auto* NSMul_data = multiplier->template data<T, Context>();
    auto* CGSMul_data = cgs_multiplier->template data<T, Context>();
    auto* NC_data = num_by_chans.template mutable_data<T, Context>();
    auto* Std_data = stddev->template mutable_data<T, Context>();
    ctx().template Copy<T, Context, Context>(Output(0)->count(), Ydata, Xdata);

    //  compute mean
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(CblasNoTrans, NG, CGS,
                       1.0 / CGS, Xdata, CGSMul_data,
                                      0, tMean_data);
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }

    //  subtract mean
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NG, CGS, 1,
                                        -1.0, tMean_data, CGSMul_data,
                                                          1.0, Ydata);
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }

    //  compute variance
    //  note that we use VAR(X) = E((X - EX) ^ 2)
    math::Square<T, Context>(Output(0)->count(), Ydata, Std_data);
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(CblasNoTrans, NG, CGS,
                    1.0 / CGS, Std_data, CGSMul_data,
                                     0.0, tVar_data);
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }

    //  compute moving average
    if (!is_recomputing) {
        //  History(X) = (1 - momentum) * Cur(X) + momentum * History(X)
        math::Axpby<T, Context>(mean->count(), 1.0 - momentum, tMean_data, momentum, hMean_data);
        math::Axpby<T, Context>(var->count(), 1.0 - momentum, tVar_data, momentum, hVar_data);
    }

    //  compute stddev
    math::AddScalar<T, Context>(var->count(), eps, tVar_data);
    math::Sqrt<T, Context>(var->count(), tVar_data, tVar_data);

    //  divide by stddev
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NG, CGS, 1,
                                          1.0, tVar_data, CGSMul_data,
                                                       0.0, Std_data);
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }
    math::Div<T, Context>(Output(0)->count(), Ydata, Std_data, Ydata);

    //  store x_norm for backward
    auto* XNorm_data = x_norm->template mutable_data<T, Context>();
    ctx().template Copy<T, Context, Context>(Output(0)->count(), XNorm_data, Ydata);

    // scale
    if (data_format == "NCHW") {
         math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, N, C, 1,
                                               1.0, NMul_data, Sdata,
                                                       0.0, NC_data);
         math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NC, S, 1,
                                              1.0, NC_data, SMul_data,
                                                       0.0, Std_data);
    } else if (data_format == "NHWC") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NS, C, 1,
                                             1.0, NSMul_data, Sdata,
                                                     0.0, Std_data);
    }
    math::Mul<T, Context>(Output(0)->count(), Ydata, Std_data, Ydata);

    // shift
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, N, C, 1,
                                             1.0, NMul_data, Bdata,
                                                     0.0, NC_data);
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NC, S, 1,
                                            1.0, NC_data, SMul_data,
                                                        1.0, Ydata);
    } else if (data_format == "NHWC") {
         math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NS, C, 1,
                                             1.0, NSMul_data,  Bdata,
                                                         1.0, Ydata);
    }
    ws()->ReleaseBuffer(stddev);
}

template <class Context> template <typename T>
void FusedGroupNormOp<Context>::InferenceRunWithType() {
    INIT_MULTIPLIER(multiplier, NS);
    INIT_MULTIPLIER(num_multiplier, N);
    INIT_MULTIPLIER(spatial_multiplier, S);
    INIT_MULTIPLIER(cgs_multiplier, CGS);
    TENSOR_FILL(Input(1), vector<TIndex>(1, NG));  //  history_mean
    TENSOR_FILL(Input(2), vector<TIndex>(1, NG));  //  history_var
    TENSOR_FILL(Input(3), vector<TIndex>(1, C));  //  scale
    TENSOR_FILL(Input(4), vector<TIndex>(1, C));  //  bias

    auto* hMean_data = Input(1).template mutable_data<T, Context>();
    auto* hVar_data = Input(2).template mutable_data<T, Context>();
    auto* Sdata = Input(3).template data<T, Context>();
    auto* Bdata = Input(4).template data<T, Context>();
    auto* tMean_data = mean->template mutable_data<T, Context>();
    auto* tVar_data = var->template mutable_data<T, Context>();
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    auto* NMul_data = num_multiplier->template data<T, Context>();
    auto* SMul_data = spatial_multiplier->template data<T, Context>();
    auto* NSMul_data = multiplier->template data<T, Context>();
    auto* CGSMul_data = cgs_multiplier->template data<T, Context>();
    auto* NC_data = num_by_chans.template mutable_data<T, Context>();
    auto* Std_data = stddev->template mutable_data<T, Context>();
    ctx().template Copy<T, Context, Context>(Input(0).count(), Ydata, Xdata);
    ctx().template Copy<T, Context, Context>(mean->count(), tMean_data, hMean_data);
    ctx().template Copy<T, Context, Context>(var->count(), tVar_data, hVar_data);

    //  subtract mean
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(CblasNoTrans, NG, CGS,
                       1.0 / CGS, Xdata, CGSMul_data,
                                      0, tMean_data);
    } else if (data_format == "NHWC") {
         NOT_IMPLEMENTED;
    }

    //  compute stddev
    math::AddScalar<T, Context>(var->count(), eps, tVar_data);
    math::Sqrt<T, Context>(var->count(), tVar_data, tVar_data);

    //  divide by stddev
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NG, CGS, 1,
                                          1.0, tVar_data, CGSMul_data,
                                                       0.0, Std_data);
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }
    math::Div<T, Context>(Output(0)->count(), Ydata, Std_data, Ydata);

    // scale
    if (data_format == "NCHW") {
         math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, N, C, 1,
                                              1.0, NMul_data, Sdata,
                                                      0.0, NC_data);
         math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NC, S, 1,
                                             1.0, NC_data, SMul_data,
                                                      0.0, Std_data);
    } else if (data_format == "NHWC") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NS, C, 1,
                                             1.0, NSMul_data, Sdata,
                                                     0.0, Std_data);
    }
    math::Mul<T, Context>(Output(0)->count(), Ydata, Std_data, Ydata);

    // shift
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, N, C, 1,
                                             1.0, NMul_data, Bdata,
                                                     0.0, NC_data);
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NC, S, 1,
                                            1.0, NC_data, SMul_data,
                                                        1.0, Ydata);
    } else if (data_format == "NHWC") {
         math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NS, C, 1,
                                             1.0, NSMul_data,  Bdata,
                                                         1.0, Ydata);
    }
    ws()->ReleaseBuffer(stddev);
}

template <class Context>
void FusedGroupNormOp<Context>::Setup() {
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
    mean = ws()->CreateTensor("/mnt/" + anchor() + "/gn_mean");
    var = ws()->CreateTensor("/mnt/" + anchor() + "/gn_var");
    x_norm = ws()->CreateTensor("/mnt/" + anchor() + "/gn_x_norm");
    stddev = ws()->GetBuffer();
    stddev->ReshapeLike(Input(0));

    //  reshape
    mean->Reshape(vector<TIndex>(1, NG));
    var->Reshape(vector<TIndex>(1, NG));
    num_by_chans.Reshape(vector<TIndex>(1, NC));
    x_norm->ReshapeLike(Input(0));
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void FusedGroupNormOp<Context>::RunOnDevice() {
    Setup();

    if (Input(0).template IsType<float>()) {
        if (use_global_stats) InferenceRunWithType<float>();
        else TrainingRunWithType<float>();
    }
#ifdef WITH_CUDA_FP16
    else if (Input(0).template IsType<float16>()) {
        if (use_global_stats) InferenceRunWithType<float16>();
        else TrainingRunWithType<float16>();
    }
#endif
    else LOG(FATAL) << "Unsupported input types.";
}


DEPLOY_CPU(FusedGroupNorm);
#ifdef WITH_CUDA
DEPLOY_CUDA(FusedGroupNorm);
#endif
OPERATOR_SCHEMA(FusedGroupNorm).NumInputs(5).NumOutputs(1);

template <class Context> template <typename T>
void FusedGroupNormGradientOp<Context>::TrainingRunWithType() {
    INIT_MULTIPLIER(multiplier, NS);
    INIT_MULTIPLIER(num_multiplier, N);
    INIT_MULTIPLIER(spatial_multiplier, S);
    INIT_MULTIPLIER(cgs_multiplier, CGS);

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Sdata = Input(3).template data<T, Context>();
    auto* Std_data = stddev->template mutable_data<T, Context>();
    auto* tMean_data = mean->template mutable_data<T, Context>();
    auto* tVar_data = var->template mutable_data<T, Context>();
    auto* NMul_data = num_multiplier->template data<T, Context>();
    auto* SMul_data = spatial_multiplier->template data<T, Context>();
    auto* NSMul_data = multiplier->template data<T, Context>();
    auto* CGSMul_data = cgs_multiplier->template data<T, Context>();
    auto* NC_data = num_by_chans.template mutable_data<T, Context>();
    auto* XNorm_data = x_norm->template data<T, Context>();

    // gradient w.r.t. scale
    if (Output(1)->name() != "ignore") {
        auto* dSdata = Output(1)->template mutable_data<T, Context>();
        math::Mul<T, Context>(stddev->count(), XNorm_data, dYdata, Std_data);
        if (data_format == "NCHW") {
            math::Gemv<T, Context>(CblasNoTrans, NC, S,
                              1.0, Std_data, SMul_data,
                                         0.0, NC_data);
            math::Gemv<T, Context>(CblasTrans, N, C,
                            1.0, NC_data, NMul_data,
                                       1.0, dSdata);
        } else if (data_format == "NHWC") {
            math::Gemv<T, Context>(CblasTrans, NS, C,
                           1.0, Std_data, NSMul_data,
                                        1.0, dSdata);
        }
    }

    // gradient w.r.t. bias
    if (Output(2)->name() != "ignore") {
        auto* dBdata = Output(2)->template mutable_data<T, Context>();
        if (data_format == "NCHW") {
            math::Gemv<T, Context>(CblasNoTrans, NC, S,
                                1.0, dYdata, SMul_data,
                                         0.0, NC_data);
            math::Gemv<T, Context>(CblasTrans, N, C,
                            1.0, NC_data, NMul_data,
                                       1.0, dBdata);
        } else if (data_format == "NHWC") {
            math::Gemv<T, Context>(CblasTrans, NS, C,
                             1.0, dYdata, NSMul_data,
                                        1.0, dBdata);
        }
    }

    // gradient w.r.t. x
    if (Output(0)->name() != "ignore") {
         // scale * dY
         if (data_format == "NCHW") {
            math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, N, C, 1,
                                                 1.0, NMul_data, Sdata,
                                                         0.0, NC_data);
            math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NC, S, 1,
                                                1.0, NC_data, SMul_data,
                                                         0.0, Std_data);
         } else if (data_format == "NHWC") {
            math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NS, C, 1,
                                                 1.0, NSMul_data, Sdata,
                                                         0.0, Std_data);
         }
         math::Mul<T, Context>(stddev->count(), Std_data, dYdata, Std_data);

         // sum of x_hat * (dl / dx_hat)
         math::Mul<T, Context>(stddev->count(), XNorm_data, Std_data, dXdata);
         if (data_format == "NCHW") {
             math::Gemv<T, Context>(CblasNoTrans, NG, CGS,
                                 1.0, dXdata, CGSMul_data,
                                         0.0, tMean_data);
         } else if (data_format == "NHWC") {
             NOT_IMPLEMENTED;
         }

         // x_hat times the sum
         if (data_format == "NCHW") {
             math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NG, CGS, 1,
                                              1.0, tMean_data, CGSMul_data,
                                                              0.0, dXdata);
         } else if (data_format == "NHWC") {
             NOT_IMPLEMENTED;
         }
         math::Mul<T, Context>(stddev->count(), XNorm_data, dXdata, dXdata);

        // subtract the average of x_hat times the sum
        if (data_format == "NCHW") {
            math::Gemv<T, Context>(CblasNoTrans, NG, CGS,
                              1.0, Std_data, CGSMul_data,
                                        0.0, tMean_data);
            math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NG, CGS, 1,
                                             1.0, tMean_data, CGSMul_data,
                                                             1.0, dXdata);
        } else if (data_format == "NHWC") {
            NOT_IMPLEMENTED;
        }
        math::Axpby<T, Context>(stddev->count(), 1.0, Std_data, -1.0 / CGS, dXdata);

        // multiply with the inverse std
         if (data_format == "NCHW") {
             math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NG, CGS, 1,
                                               1.0, tVar_data, CGSMul_data,
                                                            0.0, Std_data);
        } else if (data_format == "NHWC") {
             NOT_IMPLEMENTED;
        }
        //  divide by stddev
        math::Div<T, Context>(Output(0)->count(), dXdata, Std_data, dXdata);
    }
    ws()->ReleaseBuffer(stddev);
}

template <class Context> template <typename T>
void FusedGroupNormGradientOp<Context>::InferenceRunWithType() {
    INIT_MULTIPLIER(multiplier, NS);
    INIT_MULTIPLIER(num_multiplier, N);
    INIT_MULTIPLIER(spatial_multiplier, S);
    INIT_MULTIPLIER(cgs_multiplier, CGS);

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* Sdata = Input(3).template data<T, Context>();
    auto* tVar_data = var->template mutable_data<T, Context>();
    auto* NMul_data = num_multiplier->template data<T, Context>();
    auto* SMul_data = spatial_multiplier->template data<T, Context>();
    auto* NSMul_data = multiplier->template data<T, Context>();
    auto* CGSMul_data = cgs_multiplier->template data<T, Context>();
    auto* NC_data = num_by_chans.template mutable_data<T, Context>();

    //  gradient w.r.t. scale
    if (Output(1)->name() != "ignore") 
        LOG(FATAL) << "The gamma should be fixed if using global stats.";
       
    //  gradient w.r.t. bias
    if (Output(2)->name() != "ignore") {
        auto* dBdata = Output(2)->template mutable_data<T, Context>();
        if (data_format == "NCHW") {
            math::Gemv<T, Context>(CblasNoTrans, NC, S,
                                1.0, dYdata, SMul_data,
                                         0.0, NC_data);
            math::Gemv<T, Context>(CblasTrans, N, C,
                            1.0, NC_data, NMul_data,
                                       1.0, dBdata);
        } else if (data_format == "NHWC") {
            math::Gemv<T, Context>(CblasTrans, NS, C,
                             1.0, dYdata, NSMul_data,
                                        1.0, dBdata);
            }
    }

    //  gradient w.r.t. x
    if (Output(0)->name() != "ignore") {
        auto* dXdata = Output(0)->template mutable_data<T, Context>();
        auto* Std_data = stddev->template mutable_data<T, Context>();

        //  divide scale by stddev
        math::Div<T, Context>(var->count(), Sdata, tVar_data, tVar_data);

        //  compute dE/dY \cot (scale / std(X))
        if (data_format == "NCHW") {
            math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NG, CGS, 1,
                                              1.0, tVar_data, CGSMul_data,
                                                           0.0, Std_data);
        } else if (data_format == "NHWC") {
            NOT_IMPLEMENTED;
        }
        math::Mul<T, Context>(Output(0)->count(), dYdata, Std_data, dXdata);
    }
    ws()->ReleaseBuffer(stddev);
}

template <class Context>
void FusedGroupNormGradientOp<Context>::Setup() {
    //  determine the mode
    if (use_stats == -1) use_global_stats = phase() == "TEST" ? true : false;
    else use_global_stats = use_stats == 1 ? true : false;

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
    mean = ws()->GetTensor("/mnt/" + anchor() + "/gn_mean");
    var = ws()->GetTensor("/mnt/" + anchor() + "/gn_var");
    x_norm = ws()->GetTensor("/mnt/" + anchor() + "/gn_x_norm");
    stddev = ws()->GetBuffer();
    stddev->ReshapeLike(Input(0));

    //  reshape
    num_by_chans.Reshape(vector<TIndex>(1, NC));
    Output(0)->ReshapeLike(Input(0));  // dX
    Output(1)->ReshapeLike(Input(3));  // dScale
    Output(2)->ReshapeLike(Input(3));  // dBias
}

template <class Context>
void FusedGroupNormGradientOp<Context>::RunOnDevice() {
    Setup();

    if (Input(0).template IsType<float>()) {
        if (use_global_stats) InferenceRunWithType<float>();
        else TrainingRunWithType<float>();
    }
#ifdef WITH_CUDA_FP16
    else if (Input(0).template IsType<float16>()) {
        if (use_global_stats) InferenceRunWithType<float16>();
        else TrainingRunWithType<float16>();
    }
#endif
    else LOG(FATAL) << "Unsupported input types.";
}

template <class Context>
void FusedGroupNormGradientOp<Context>::ShareGradient() {
    if (use_global_stats) {
        if (Output(0)->name() != "ignore") {
            Tensor* dX = ws()->GetBuffer("Grad");
            ws()->CreateAvatar(Output(0), dX);
        }
    } else {
        if (Output(0)->name() != "ignore" ||
            Output(1)->name() != "ignore" ||
            Output(2)->name() != "ignore") {
            Tensor* dX = ws()->GetBuffer("Grad");
            ws()->CreateAvatar(Output(0), dX);
        }
    }
}

DEPLOY_CPU(FusedGroupNormGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(FusedGroupNormGradient);
#endif
OPERATOR_SCHEMA(FusedGroupNormGradient).NumInputs(5).NumOutputs(3);

class GetFusedGroupNormGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetFusedGroupNormGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), I(2), I(3), GO(0)},
            vector<string> {GI(0), GI(3), GI(4)});
    }
};
REGISTER_GRADIENT(FusedGroupNorm, GetFusedGroupNormGradient);

}    // namespace dragon