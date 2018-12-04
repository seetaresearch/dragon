#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/math_functions.h"
#include "operators/norm/instance_norm_op.h"

namespace dragon {

template <class Context> template <typename T>
void InstanceNormOp<Context>::RunWithType() {
    DECLARE_MULTIPLIER(Smult, S);

    auto* Tmean = mean.template mutable_data<T, Context>();
    auto* Tvar = var->template mutable_data<T, Context>();
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    auto* WSdata = ws()->template caches<T, Context>({ Input(0).count() })[0];
    ctx()->template Copy<T, Context, Context>(Output(0)->count(), Ydata, Xdata);

    // Compute mean
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(
            CblasNoTrans, NC, S,
                1.f / S, Xdata, Smult,
                    0.f, Tmean, ctx());
    } else if (data_format == "NHWC") {
        auto* x = Xdata;
        auto* tm = Tmean;
        for (int i = 0; i < N; i++) {
            math::Gemv<T, Context>(
                CblasTrans, S, C,
                    1.f / S, x, Smult,
                        0.f, tm, ctx());
            x += CS;
            tm += C;
        }
    }

    // Subtract mean
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NC, S, 1,
                    -1.f, Tmean, Smult,
                        1.f, Ydata, ctx());
    } else if (data_format == "NHWC") {
        auto* y = Ydata;
        auto* tm = Tmean;
        for (int i = 0; i < N; i++) {
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    S, C, 1,
                        -1.f, Smult, tm,
                            1.f, y, ctx());
            y += CS;
            tm += C;
        }
    }
  
    // Compute variance
    // Note that we use VAR(X) = E((X - EX) ^ 2)
    math::Square<T, Context>(Output(0)->count(), Ydata, WSdata, ctx());
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(
            CblasNoTrans, NC, S,
                1.f / S, WSdata, Smult,
                    0.f, Tvar, ctx());
    } else if (data_format == "NHWC") {
        auto* x2 = WSdata;
        auto* tv = Tvar;
        for (int i = 0; i < N; i++) {
            math::Gemv<T, Context>(
                CblasTrans, S, C,
                    1.f / S, x2, Smult,
                        0.f, tv, ctx());
            x2 += CS;
            tv += C;
        }
    }

    // Compute stddev
    math::AddScalar<T, Context>(var->count(), eps, Tvar, ctx());
    math::Sqrt<T, Context>(var->count(), Tvar, Tvar, ctx());

    // Divide by stddev
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NC, S, 1,
                    1.f, Tvar, Smult,
                        0.f, WSdata, ctx());
    } else if (data_format == "NHWC") {
        auto* std = WSdata;
        auto* tv = Tvar;
        for (int i = 0; i < N; i++) {
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    S, C, 1,
                        1.f, Smult, tv,
                            0.f, std, ctx());
            std += CS;
            tv += C;
        }
    }
    math::Div<T, Context>(Output(0)->count(),
        Ydata, WSdata, Ydata, ctx());
}

template <class Context>
void InstanceNormOp<Context>::Setup() {
    // Determine the data format
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

    // Make resource
    var = ws()->CreateTensor("/mnt/" + anchor() + "/ins_norm/var");

    // Reshape
    mean.Reshape({ NC });
    var->Reshape({ NC });
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void InstanceNormOp<Context>::RunOnDevice() {
    Setup();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(InstanceNorm);
#ifdef WITH_CUDA
DEPLOY_CUDA(InstanceNorm);
#endif
OPERATOR_SCHEMA(InstanceNorm).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void InstanceNormGradientOp<Context>::RunWithType() {
    DECLARE_MULTIPLIER(Smult, S);

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Tvar = var->template mutable_data<T, Context>();
    auto* WSdata = ws()->template caches<T, Context>({ Output(0)->count() })[0];

    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NC, S, 1,
                    1.f, Tvar, Smult,
                        0.f, WSdata, ctx());
    } else if (data_format == "NHWC") {
        auto* std = WSdata;
        auto* tv = Tvar;
        for (int i = 0; i < N; i++) {
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    S, C, 1,
                        1.f, Smult, tv,
                            0.f, std, ctx());
            std += CS;
            tv += C;
        }
    }

    auto* Ydata = Input(-2).template data<T, Context>();
    math::Mul<T, Context>(Output(0)->count(),
        Ydata, dYdata, dXdata, ctx());

    // sum(dE/dY \cdot Y)
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(
            CblasNoTrans, NC, S,
                1.f, dXdata, Smult,
                    0.f, Tvar, ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NC, S, 1,
                    1.f, Tvar, Smult,
                        0.f, dXdata, ctx());
    } else if (data_format == "NHWC") {
        for (int i = 0; i < N; i++) {
            auto* dx = dXdata;
            auto* tv = Tvar;
            for (int i = 0; i < N; i++) {
                math::Gemv<T, Context>(
                    CblasTrans, S, C,
                        1.f, dx, Smult,
                            0.f, tv, ctx());
                math::Gemm<T, Context>(
                    CblasNoTrans, CblasNoTrans,
                        S, C, 1,
                            1.f, Smult, tv,
                                0.f, dx, ctx());
                dx += CS;
                tv += C;
            }
        }
    }

    // sum(dE/dY \cdot Y) \cdot Y
    math::Mul<T, Context>(Output(0)->count(),
        Ydata, dXdata, dXdata, ctx());

    // sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(
            CblasNoTrans, NC, S,
                1.f, dYdata, Smult,
                    0.f, Tvar, ctx());
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                NC, S, 1,
                    1.f, Tvar, Smult,
                        1.f, dXdata, ctx());
    } else if (data_format == "NHWC") {
        for (int i = 0; i < N; i++) {
            auto* dy = dYdata;
            auto* dx = dXdata;
            auto* tv = Tvar;
            for (int i = 0; i < N; i++) {
                math::Gemv<T, Context>(
                    CblasTrans, S, C,
                        1.f, dy, Smult,
                            0.f, tv, ctx());
                math::Gemm<T, Context>(
                    CblasNoTrans, CblasNoTrans,
                        S, C, 1,
                            1.f, Smult, tv,
                                1.f, dx, ctx());
                dy += CS;
                dx += CS;
                tv += C;
            }
        }
    }
   
    //   dE/dY - mean(dE/dY)- mean(dE/dY \cdot Y) \cdot Y
    // = dE/dY - mean(sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y)
    math::Axpby<T, Context>(Output(0)->count(),
        1.f, dYdata, -1.f / S, dXdata, ctx());

    // Divide by stddev
    math::Div<T, Context>(Output(0)->count(),
        dXdata, WSdata, dXdata, ctx());
}

template <class Context>
void InstanceNormGradientOp<Context>::Setup() {
    // Determine the data format
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

    // Make resource
    var = ws()->GetTensor("/mnt/" + anchor() + "/ins_norm/var");

    // Reshape
    Output(0)->ReshapeLike(Input(0));
}


template <class Context>
void InstanceNormGradientOp<Context>::RunOnDevice() {
    Setup();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
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

}  // namespace dragon