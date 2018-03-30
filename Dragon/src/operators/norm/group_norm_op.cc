#include "operators/norm/group_norm_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/filler.h"

namespace dragon {

template <class Context> template <typename T>
void GroupNormOp<Context>::TrainingRunWithType() {
    INIT_MULTIPLIER(multiplier, NS);
    INIT_MULTIPLIER(num_multiplier, N);
    INIT_MULTIPLIER(spatial_multiplier, S);
    INIT_MULTIPLIER(cgs_multiplier, CGS);
    TENSOR_FILL(Input(1), vector<TIndex>(1, NG));  //  history_mean
    TENSOR_FILL(Input(2), vector<TIndex>(1, NG));  //  history_var

    auto* hMean_data = Input(1).template mutable_data<T, Context>();
    auto* hVar_data = Input(2).template mutable_data<T, Context>();
    auto* tMean_data = mean.template mutable_data<T, Context>();
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
            math::Axpby<T, Context>(mean.count(), 1.0, tMean_data, momentum, hMean_data);
            math::Axpby<T, Context>(var->count(), coeff, tVar_data, momentum, hVar_data);
        } else {
            //  History(X) = (1 - momentum) * Cur(X) + momentum * History(X)
            math::Axpby<T, Context>(mean.count(), 1.0 - momentum, tMean_data, momentum, hMean_data);
            math::Axpby<T, Context>(var->count(), 1.0 - momentum, tVar_data, momentum, hVar_data);
        }
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
    ws()->ReleaseBuffer(stddev);
}

template <class Context> template <typename T>
void GroupNormOp<Context>::InferenceRunWithType() {
    INIT_MULTIPLIER(multiplier, NS);
    INIT_MULTIPLIER(num_multiplier, N);
    INIT_MULTIPLIER(spatial_multiplier, S);
    INIT_MULTIPLIER(cgs_multiplier, CGS);
    TENSOR_FILL(Input(1), vector<TIndex>(1, NG));  //  history_mean
    TENSOR_FILL(Input(2), vector<TIndex>(1, NG));  //  history_var

    auto* hMean_data = Input(1).template mutable_data<T, Context>();
    auto* hVar_data = Input(2).template mutable_data<T, Context>();
    auto* tMean_data = mean.template mutable_data<T, Context>();
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

    //  scale the mean and variance if necessary
    if (mode == "CAFFE") {
        CHECK_EQ(InputSize(), 4)
            << "\nThe number of inputs should be 4 if use CAFFE mode.";
        TENSOR_FILL(Input(3), vector<TIndex>(1, 1));
        auto* hFact_data = Input(3).template mutable_data<T, CPUContext>();
        const float factor = dragon_cast<float, T>(hFact_data[0]);
        const float scale = factor == 0 ? 0 : 1.0 / factor;
        math::Scale<T, Context>(mean.count(), scale, hMean_data, tMean_data);
        math::Scale<T, Context>(var->count(), scale, hVar_data, tVar_data);
    } else {
       ctx().template Copy<T, Context, Context>(mean.count(), tMean_data, hMean_data);
       ctx().template Copy<T, Context, Context>(var->count(), tVar_data, hVar_data);
    }

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
    ws()->ReleaseBuffer(stddev);
}

template <class Context>
void GroupNormOp<Context>::Setup() {
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
    var = ws()->CreateTensor("/mnt/" + Anchor() + "/gn/var");
    stddev = ws()->GetBuffer();
    stddev->ReshapeLike(Input(0));

    //  reshape
    mean.Reshape(vector<TIndex>(1, NG));
    var->Reshape(vector<TIndex>(1, NG));
    num_by_chans.Reshape(vector<TIndex>(1, NC));
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void GroupNormOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(GroupNorm);
#ifdef WITH_CUDA
DEPLOY_CUDA(GroupNorm);
#endif
OPERATOR_SCHEMA(GroupNorm).NumInputs(3, 4).NumOutputs(1);

template <class Context> template <typename T>
void GroupNormGradientOp<Context>::TrainingRunWithType() {
    INIT_MULTIPLIER(multiplier, NS);
    INIT_MULTIPLIER(num_multiplier, N);
    INIT_MULTIPLIER(spatial_multiplier, S);
    INIT_MULTIPLIER(cgs_multiplier, CGS);

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Std_data = stddev->template mutable_data<T, Context>();
    auto* tVar_data = var->template mutable_data<T, Context>();
    auto* NMul_data = num_multiplier->template data<T, Context>();
    auto* SMul_data = spatial_multiplier->template data<T, Context>();
    auto* NSMul_data = multiplier->template data<T, Context>();
    auto* CGSMul_data = cgs_multiplier->template data<T, Context>();
    auto* NC_data = num_by_chans.template mutable_data<T, Context>();

    if (data_format == "NCHW") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NG, CGS, 1,
                                          1.0, tVar_data, CGSMul_data,
                                                       0.0, Std_data);
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }

    auto* Ydata = Input(1).template data<T, Context>();
    math::Mul<T, Context>(Output(0)->count(), Ydata, dYdata, dXdata);

     //  sum(dE/dY \cdot Y)
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(CblasNoTrans, NG, CGS,
                            1.0, dXdata, CGSMul_data,
                                     0.0, tVar_data);
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NG, CGS, 1,
                                          1.0, tVar_data, CGSMul_data,
                                                         0.0, dXdata);
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }

    //  sum(dE/dY \cdot Y) \cdot Y
    math::Mul<T, Context>(Output(0)->count(), Ydata, dXdata, dXdata);

    //  sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(CblasNoTrans, NG, CGS,
                            1.0, dYdata, CGSMul_data,
                                     0.0, tVar_data);
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NG, CGS, 1,
                                          1.0, tVar_data, CGSMul_data,
                                                         1.0, dXdata);
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }

    //   dE/dY - mean(dE/dY)- mean(dE/dY \cdot Y) \cdot Y
    // = dE/dY - mean(sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y)
    math::Axpby<T, Context>(Output(0)->count(), 1.0, dYdata, -1.0 / CGS, dXdata);

    //  divide by stddev
    math::Div<T, Context>(Output(0)->count(), dXdata, Std_data, dXdata);
    ws()->ReleaseBuffer(stddev);
}

template <class Context> template <typename T>
void GroupNormGradientOp<Context>::InferenceRunWithType() {
    INIT_MULTIPLIER(multiplier, NS);
    INIT_MULTIPLIER(num_multiplier, N);
    INIT_MULTIPLIER(spatial_multiplier, S);
    INIT_MULTIPLIER(cgs_multiplier, CGS);

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Std_data = stddev->template mutable_data<T, Context>();
    auto* tVar_data = var->template mutable_data<T, Context>();
    auto* NMul_data = num_multiplier->template data<T, Context>();
    auto* SMul_data = spatial_multiplier->template data<T, Context>();
    auto* NSMul_data = multiplier->template data<T, Context>();
    auto* CGSMul_data = cgs_multiplier->template data<T, Context>();
    auto* NC_data = num_by_chans.template mutable_data<T, Context>();

    if (data_format == "NCHW") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NG, CGS, 1,
                                           1.0, tVar_data, NSMul_data,
                                                       0.0, Std_data);
    } else if (data_format == "NHWC") {
        NOT_IMPLEMENTED;
    }

    math::Div<T, Context>(Output(0)->count(), dYdata, Std_data, dXdata);
    ws()->ReleaseBuffer(stddev);
}

template <class Context>
void GroupNormGradientOp<Context>::Setup() {
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
    var = ws()->GetTensor("/mnt/" + Anchor() + "/gn/var");
    stddev = ws()->GetBuffer();
    stddev->ReshapeLike(Input(0));

    //  reshape
    num_by_chans.Reshape(vector<TIndex>(1, NC));
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void GroupNormGradientOp<Context>::RunOnDevice() {
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