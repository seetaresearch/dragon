#include "operators/norm/group_norm_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/filler.h"

namespace dragon {

template <class Context> template <typename T>
void GroupNormOp<Context>::RunWithType() {
    INIT_MULTIPLIER(multiplier, NS);
    INIT_MULTIPLIER(num_multiplier, N);
    INIT_MULTIPLIER(spatial_multiplier, S);
    INIT_MULTIPLIER(cgs_multiplier, CGS);

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
    var = ws()->CreateTensor("/mnt/" + anchor() + "/gn/var");
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
    if (group == C && Input(0).ndim() == 2)    //  InstanceNorm
        LOG(WARNING) << "The 2d input will output all zeros.";
    NC = N * C;
    NG = N * group;
    S = Input(0).count() / NC;
    CGS = (C / group) * S;
    NS = N * S;

    //  make resource
    var = ws()->GetTensor("/mnt/" + anchor() + "/gn/var");
    stddev = ws()->GetBuffer();
    stddev->ReshapeLike(Input(0));

    //  reshape
    num_by_chans.Reshape(vector<TIndex>(1, NC));
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