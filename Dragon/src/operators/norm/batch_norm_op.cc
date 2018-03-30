#include "operators/norm/batch_norm_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/filler.h"

namespace dragon {

template <class Context> template <typename T>
void BatchNormOp<Context>::TrainingRunWithType() {
    INIT_MULTIPLIER(multiplier, NS);
    INIT_MULTIPLIER(num_multiplier, N);
    INIT_MULTIPLIER(spatial_multiplier, S);
    TENSOR_FILL(Input(1), vector<TIndex>(1, C));  //  history_mean
    TENSOR_FILL(Input(2), vector<TIndex>(1, C));  //  history_var

    auto* hMean_data = Input(1).template mutable_data<T, Context>();
    auto* hVar_data = Input(2).template mutable_data<T, Context>();
    auto* tMean_data = mean.template mutable_data<T, Context>();
    auto* tVar_data = var->template mutable_data<T, Context>();
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    auto* NMul_data = num_multiplier->template data<T, Context>();
    auto* SMul_data = spatial_multiplier->template data<T, Context>();
    auto* NSMul_data = multiplier->template data<T, Context>();
    auto* NC_data = num_by_chans.template mutable_data<T, Context>();
    auto* Std_data = stddev->template mutable_data<T, Context>();
    ctx().template Copy<T, Context, Context>(Output(0)->count(), Ydata, Xdata);

    //  compute mean
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(CblasNoTrans, NC, S,
                        1.0 / NS, Xdata, SMul_data,
                                       0, NC_data);
        math::Gemv<T, Context>(CblasTrans, N, C,
                        1.0, NC_data, NMul_data,
                                 0, tMean_data);
    } else if (data_format == "NHWC") {
        math::Gemv<T, Context>(CblasTrans, NS, C,
                     1.0 / NS, Xdata, NSMul_data,
                                  0, tMean_data);
    }

    //  subtract mean
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, N, C, 1,
                                        1.0, NMul_data, tMean_data,
                                                     0.0, NC_data);
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NC, S, 1,
                                           -1.0, NC_data, SMul_data,
                                                        1.0, Ydata);
    } else if (data_format == "NHWC") {
         math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NS, C, 1,
                                        -1.0, NSMul_data, tMean_data,
                                                         1.0, Ydata);
    }

    //  compute variance
    //  note that we use VAR(X) = E((X - EX) ^ 2)
    math::Square<T, Context>(Output(0)->count(), Ydata, Std_data);
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(CblasNoTrans, NC, S,
                   1.0 / NS, Std_data, SMul_data,
                                     0.0, NC_data);
        math::Gemv<T, Context>(CblasTrans, N, C,
                        1.0, NC_data, NMul_data,
                                0.0, tVar_data);
    } else if (data_format == "NHWC") {
        math::Gemv<T, Context>(CblasTrans, NS, C,
                  1.0 / NS, Std_data, NSMul_data,
                                 0.0, tVar_data);
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
          math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, N, C, 1,
                                           1.0, NMul_data, tVar_data,
                                                       0.0, NC_data);
          math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NC, S, 1,
                                              1.0, NC_data, SMul_data,
                                                       0.0, Std_data);
    } else if (data_format == "NHWC") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NS, C, 1,
                                         1.0, NSMul_data, tVar_data,
                                                     0.0, Std_data);
    }
    math::Div<T, Context>(Output(0)->count(), Ydata, Std_data, Ydata);
    ws()->ReleaseBuffer(stddev);
}

template <class Context> template <typename T>
void BatchNormOp<Context>::InferenceRunWithType() {
    INIT_MULTIPLIER(multiplier, NS);
    INIT_MULTIPLIER(num_multiplier, N);
    INIT_MULTIPLIER(spatial_multiplier, S);
    TENSOR_FILL(Input(1), vector<TIndex>(1, C));  //  history_mean
    TENSOR_FILL(Input(2), vector<TIndex>(1, C));  //  history_var

    auto* hMean_data = Input(1).template mutable_data<T, Context>();
    auto* hVar_data = Input(2).template mutable_data<T, Context>();
    auto* tMean_data = mean.template mutable_data<T, Context>();
    auto* tVar_data = var->template mutable_data<T, Context>();
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    auto* NMul_data = num_multiplier->template data<T, Context>();
    auto* SMul_data = spatial_multiplier->template data<T, Context>();
    auto* NSMul_data = multiplier->template data<T, Context>();
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
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, N, C, 1,
                                        1.0, NMul_data, tMean_data,
                                                     0.0, NC_data);
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NC, S, 1,
                                           -1.0, NC_data, SMul_data,
                                                        1.0, Ydata);
    } else if (data_format == "NHWC") {
         math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NS, C, 1,
                                        -1.0, NSMul_data, tMean_data,
                                                         1.0, Ydata);
    }

    //  compute stddev
    math::AddScalar<T, Context>(var->count(), eps, tVar_data);
    math::Sqrt<T, Context>(var->count(), tVar_data, tVar_data);

    //  divide by stddev
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, N, C, 1,
                                         1.0, NMul_data, tVar_data,
                                                     0.0, NC_data);
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NC, S, 1,
                                            1.0, NC_data, SMul_data,
                                                     0.0, Std_data);
    } else if (data_format == "NHWC") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NS, C, 1,
                                         1.0, NSMul_data, tVar_data,
                                                     0.0, Std_data);
    }
    math::Div<T, Context>(Output(0)->count(), Ydata, Std_data, Ydata);
    ws()->ReleaseBuffer(stddev);
}

template <class Context>
void BatchNormOp<Context>::Setup() {
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
    var = ws()->CreateTensor("/mnt/" + Anchor() + "/bn/var");
    stddev = ws()->GetBuffer();
    stddev->ReshapeLike(Input(0));

    //  reshape
    mean.Reshape(vector<TIndex>(1, C));
    var->Reshape(vector<TIndex>(1, C));
    num_by_chans.Reshape(vector<TIndex>(1, NC));
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void BatchNormOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(BatchNorm);
#ifdef WITH_CUDA
DEPLOY_CUDA(BatchNorm);
#endif
OPERATOR_SCHEMA(BatchNorm).NumInputs(3, 4).NumOutputs(1);

template <class Context> template <typename T>
void BatchNormGradientOp<Context>::TrainingRunWithType() {
    INIT_MULTIPLIER(multiplier, NS);
    INIT_MULTIPLIER(num_multiplier, N);
    INIT_MULTIPLIER(spatial_multiplier, S);

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Std_data = stddev->template mutable_data<T, Context>();
    auto* tVar_data = var->template mutable_data<T, Context>();
    auto* NMul_data = num_multiplier->template data<T, Context>();
    auto* SMul_data = spatial_multiplier->template data<T, Context>();
    auto* NSMul_data = multiplier->template data<T, Context>();
    auto* NC_data = num_by_chans.template mutable_data<T, Context>();

    if (data_format == "NCHW") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, N, C, 1,
                                         1.0, NMul_data, tVar_data,
                                                     0.0, NC_data);
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NC, S, 1,
                                            1.0, NC_data, SMul_data,
                                                     0.0, Std_data);
    } else if (data_format == "NHWC") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NS, C, 1,
                                         1.0, NSMul_data, tVar_data,
                                                     0.0, Std_data);
    }

    auto* Ydata = Input(1).template data<T, Context>();
    math::Mul<T, Context>(Output(0)->count(), Ydata, dYdata, dXdata);

     //  sum(dE/dY \cdot Y)
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(CblasNoTrans, NC, S,
                            1.0, dXdata, SMul_data,
                                     0.0, NC_data);
        math::Gemv<T, Context>(CblasTrans, N, C,
                        1.0, NC_data, NMul_data,
                                0.0, tVar_data);
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, N, C, 1,
                                         1.0, NMul_data, tVar_data,
                                                     0.0, NC_data);
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NC, S, 1,
                                            1.0, NC_data, SMul_data,
                                                       0.0, dXdata);
    } else if (data_format == "NHWC") {
        math::Gemv<T, Context>(CblasTrans, NS, C,
                         1.0, dXdata, NSMul_data,
                                 0.0, tVar_data);
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NS, C, 1,
                                         1.0, NSMul_data, tVar_data,
                                                       0.0, dXdata);
    }

    //  sum(dE/dY \cdot Y) \cdot Y
    math::Mul<T, Context>(Output(0)->count(), Ydata, dXdata, dXdata);

    //  sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(CblasNoTrans, NC, S,
                            1.0, dYdata, SMul_data,
                                     0.0, NC_data);
        math::Gemv<T, Context>(CblasTrans, N, C,
                        1.0, NC_data, NMul_data,
                                0.0, tVar_data);
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, N, C, 1,
                                         1.0, NMul_data, tVar_data,
                                                     0.0, NC_data);
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NC, S, 1,
                                            1.0, NC_data, SMul_data,
                                                       1.0, dXdata);
    } else if (data_format == "NHWC") {
        math::Gemv<T, Context>(CblasTrans, NS, C,
                         1.0, dYdata, NSMul_data,
                                 0.0, tVar_data);
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NS, C, 1,
                                         1.0, NSMul_data, tVar_data,
                                                       1.0, dXdata);
    }

    //   dE/dY - mean(dE/dY)- mean(dE/dY \cdot Y) \cdot Y
    // = dE/dY - mean(sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y)
    math::Axpby<T, Context>(Output(0)->count(), 1.0, dYdata, -1.0 / NS, dXdata);

    //  divide by stddev
    math::Div<T, Context>(Output(0)->count(), dXdata, Std_data, dXdata);
    ws()->ReleaseBuffer(stddev);
}

template <class Context> template <typename T>
void BatchNormGradientOp<Context>::InferenceRunWithType() {
    INIT_MULTIPLIER(multiplier, NS);
    INIT_MULTIPLIER(num_multiplier, N);
    INIT_MULTIPLIER(spatial_multiplier, S);

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Std_data = stddev->template mutable_data<T, Context>();
    auto* tVar_data = var->template mutable_data<T, Context>();
    auto* NMul_data = num_multiplier->template data<T, Context>();
    auto* SMul_data = spatial_multiplier->template data<T, Context>();
    auto* NSMul_data = multiplier->template data<T, Context>();
    auto* NC_data = num_by_chans.template mutable_data<T, Context>();

    if (data_format == "NCHW") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, N, C, 1,
                                         1.0, NMul_data, tVar_data,
                                                     0.0, NC_data);
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NC, S, 1,
                                            1.0, NC_data, SMul_data,
                                                     0.0, Std_data);
    } else if (data_format == "NHWC") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NS, C, 1,
                                         1.0, NSMul_data, tVar_data,
                                                     0.0, Std_data);
    }

    math::Div<T, Context>(Output(0)->count(), dYdata, Std_data, dXdata);
    ws()->ReleaseBuffer(stddev);
}

template <class Context>
void BatchNormGradientOp<Context>::Setup() {
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
    var = ws()->GetTensor("/mnt/" + Anchor() + "/bn/var");
    stddev = ws()->GetBuffer();
    stddev->ReshapeLike(Input(0));

    //  reshape
    num_by_chans.Reshape(vector<TIndex>(1, NC));
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void BatchNormGradientOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(BatchNormGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(BatchNormGradient);
#endif
OPERATOR_SCHEMA(BatchNormGradient).NumInputs(3).NumOutputs(1);

class GetBatchNormGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetBatchNormGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), O(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(BatchNorm, GetBatchNormGradient);

}    // namespace dragon