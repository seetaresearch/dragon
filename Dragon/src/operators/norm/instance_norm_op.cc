#include "operators/norm/instance_norm_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/filler.h"

namespace dragon {

template <class Context> template <typename T>
void InstanceNormOp<Context>::RunWithType() {
    INIT_MULTIPLIER(spatial_multiplier, S);

    auto* tMean_data = mean.template mutable_data<T, Context>();
    auto* tVar_data = var->template mutable_data<T, Context>();
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    auto* SMul_data = spatial_multiplier->template data<T, Context>();
    auto* Std_data = stddev->template mutable_data<T, Context>();
    ctx().template Copy<T, Context, Context>(Output(0)->count(), Ydata, Xdata);

    //  compute mean
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(CblasNoTrans, NC, S,
                         1.0 / S, Xdata, SMul_data,
                                    0, tMean_data);
    } else if (data_format == "NHWC") {
        auto* x = Xdata;
        auto* tm = tMean_data;
        for (int i = 0; i < N; i++) {
            math::Gemv<T, Context>(CblasTrans, S, C,
                              1.0 / S, x, SMul_data,
                                             0, tm);
            x += CS;
            tm += C;
        }
    }

    //  subtract mean
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NC, S, 1,
                                        -1.0, tMean_data, SMul_data,
                                                        1.0, Ydata);
    } else if (data_format == "NHWC") {
        auto* y = Ydata;
        auto* tm = tMean_data;
        for (int i = 0; i < N; i++) {
            math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, S, C, 1,
                                                   -1.0, SMul_data, tm,
                                                               1.0, y);
            y += CS;
            tm += C;
        }
    }
  
    //  compute variance
    //  note that we use VAR(X) = E((X - EX) ^ 2)
    math::Square<T, Context>(Output(0)->count(), Ydata, Std_data);
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(CblasNoTrans, NC, S,
                      1.0 / S, Std_data, SMul_data,
                                   0.0, tVar_data);
    } else if (data_format == "NHWC") {
        auto* x2 = Std_data;
        auto* tv = tVar_data;
        for (int i = 0; i < N; i++) {
            math::Gemv<T, Context>(CblasTrans, S, C,
                             1.0 / S, x2, SMul_data,
                                             0, tv);
            x2 += CS;
            tv += C;
        }
    }

    //  compute stddev
    math::AddScalar<T, Context>(var->count(), eps, tVar_data);
    math::Sqrt<T, Context>(var->count(), tVar_data, tVar_data);

    //  divide by stddev
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NC, S, 1,
                                          1.0, tVar_data, SMul_data,
                                                     0.0, Std_data);
    } else if (data_format == "NHWC") {
        auto* std = Std_data;
        auto* tv = tVar_data;
        for (int i = 0; i < N; i++) {
            math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, S, C, 1,
                                                    1.0, SMul_data, tv,
                                                             0.0, std);
            std += CS;
            tv += C;
        }
    }
    math::Div<T, Context>(Output(0)->count(), Ydata, Std_data, Ydata);
    ws()->ReleaseBuffer(stddev);
}

template <class Context>
void InstanceNormOp<Context>::Setup() {
    //  determine the data format
    TIndex channel_axis = axis;
    data_format = "NCHW";
    if (channel_axis == -1) channel_axis += (int)Input(0).ndim();
    if (channel_axis + 1 == (int)Input(0).ndim()) data_format = "NHWC";
    if (Input(0).ndim() == 2) LOG(WARNING) << "The 2d input will output all zeros.";

    N = Input(0).dim(0);
    C = Input(0).dim(channel_axis);
    NC = N * C;
    S = Input(0).count() / NC;
    CS = C * S;

    //  make resource
    var = ws()->CreateTensor("/mnt/" + anchor() + "/ins_norm/var");
    stddev = ws()->GetBuffer();
    stddev->ReshapeLike(Input(0));

    //  reshape
    mean.Reshape(vector<TIndex>(1, NC));
    var->Reshape(vector<TIndex>(1, NC));
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void InstanceNormOp<Context>::RunOnDevice() {
    Setup();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(InstanceNorm);
#ifdef WITH_CUDA
DEPLOY_CUDA(InstanceNorm);
#endif
OPERATOR_SCHEMA(InstanceNorm).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void InstanceNormGradientOp<Context>::RunWithType() {
    INIT_MULTIPLIER(spatial_multiplier, S);

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Std_data = stddev->template mutable_data<T, Context>();
    auto* tVar_data = var->template mutable_data<T, Context>();
    auto* SMul_data = spatial_multiplier->template data<T, Context>();

    if (data_format == "NCHW") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NC, S, 1,
                                          1.0, tVar_data, SMul_data,
                                                     0.0, Std_data);
    } else if (data_format == "NHWC") {
        auto* std = Std_data;
        auto* tv = tVar_data;
        for (int i = 0; i < N; i++) {
            math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, S, C, 1,
                                                    1.0, SMul_data, tv,
                                                             0.0, std);
            std += CS;
            tv += C;
        }
    }

    auto* Ydata = Input(-2).template data<T, Context>();
    math::Mul<T, Context>(Output(0)->count(), Ydata, dYdata, dXdata);

    //  sum(dE/dY \cdot Y)
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(CblasNoTrans, NC, S,
                            1.0, dXdata, SMul_data,
                                   0.0, tVar_data);
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NC, S, 1,
                                          1.0, tVar_data, SMul_data,
                                                       0.0, dXdata);
    } else if (data_format == "NHWC") {
        for (int i = 0; i < N; i++) {
            auto* dx = dXdata;
            auto* tv = tVar_data;
            for (int i = 0; i < N; i++) {
                math::Gemv<T, Context>(CblasTrans, S, C,
                                     1.0, dx, SMul_data,
                                                 0, tv);
                math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, S, C, 1,
                                                        1.0, SMul_data, tv,
                                                                  0.0, dx);
                dx += CS;
                tv += C;
            }
        }
    }

    //  sum(dE/dY \cdot Y) \cdot Y
    math::Mul<T, Context>(Output(0)->count(), Ydata, dXdata, dXdata);

    //  sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(CblasNoTrans, NC, S,
                            1.0, dYdata, SMul_data,
                                   0.0, tVar_data);
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, NC, S, 1,
                                          1.0, tVar_data, SMul_data,
                                                       1.0, dXdata);
    } else if (data_format == "NHWC") {
        for (int i = 0; i < N; i++) {
            auto* dy = dYdata;
            auto* dx = dXdata;
            auto* tv = tVar_data;
            for (int i = 0; i < N; i++) {
                math::Gemv<T, Context>(CblasTrans, S, C,
                                     1.0, dy, SMul_data,
                                                 0, tv);
                math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, S, C, 1,
                                                        1.0, SMul_data, tv,
                                                                  1.0, dx);
                dy += CS;
                dx += CS;
                tv += C;
            }
        }
    }
   
    //  dE/dY - mean(dE/dY)- mean(dE/dY \cdot Y) \cdot Y
    //  = dE/dY - mean(sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y)
    math::Axpby<T, Context>(Output(0)->count(), 1.0, dYdata, -1.0 / S, dXdata);

    //  divide by stddev
    math::Div<T, Context>(Output(0)->count(), dXdata, Std_data, dXdata);
    ws()->ReleaseBuffer(stddev);
}

template <class Context>
void InstanceNormGradientOp<Context>::Setup() {
    //  determine the data format
    TIndex channel_axis = axis;
    data_format = "NCHW";
    if (channel_axis == -1) channel_axis += (int)Input(0).ndim();
    if (channel_axis + 1 == (int)Input(0).ndim()) data_format = "NHWC";
    if (Input(0).ndim() == 2) LOG(WARNING) << "The 2d input will output all zeros.";

    N = Input(0).dim(0);
    C = Input(0).dim(channel_axis);
    NC = N * C;
    S = Input(0).count() / NC;
    CS = C * S;

    //  make resource
    var = ws()->GetTensor("/mnt/" + anchor() + "/ins_norm/var");
    stddev = ws()->GetBuffer();
    stddev->ReshapeLike(Input(0));

    //  reshape
    Output(0)->ReshapeLike(Input(0));
}


template <class Context>
void InstanceNormGradientOp<Context>::RunOnDevice() {
    Setup();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(InstanceNormGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(InstanceNormGradient);
#endif
OPERATOR_SCHEMA(InstanceNormGradient).NumInputs(3).NumOutputs(1);

class GetInstanceNormGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetInstanceNormGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), O(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(InstanceNorm, GetInstanceNormGradient);

}    // namespace dragon