#include "operators/norm/instance_norm_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/filler.h"

namespace dragon {

template <class Context> template <typename T>
void InstanceNormOp<Context>::RunWithType() {
    INIT_MULTIPLIER(spatial_multiplier, spatial_dim);

    //  get buffer
    stddev = ws()->GetBuffer();
    stddev->ReshapeLike(input(0));

    auto* tMean_data = mean.template mutable_data<T, Context>();
    auto* tVar_data = var->template mutable_data<T, Context>();
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    auto* SMul_data = spatial_multiplier->template data<T, Context>();
    auto* Std_data = stddev->template mutable_data<T, Context>();

    math::Gemv<T, Context>(CblasNoTrans, nbychans, spatial_dim,
                                             1.0 / spatial_dim, 
                                              Xdata, SMul_data, 
                                                0, tMean_data);

    if (!inplace) {
        ctx().template Copy<T, Context, Context>(input(0).count(), Ydata, Xdata);
    }

    //  subtract mean
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, nbychans, spatial_dim, 1, 
                                                                           -1.0, 
                                                          tMean_data, SMul_data, 
                                                                    1.0, Ydata);

    //  Var(X) = E((X - EX) ^ 2)
    math::Pow<T, Context>(output(0)->count(), 2, Ydata, Std_data);
    math::Gemv<T, Context>(CblasNoTrans, nbychans, spatial_dim,
                                             1.0 / spatial_dim, 
                                           Std_data, SMul_data, 
                                                           0.0, 
                                                    tVar_data);

    //  normalize var
    math::AddScalar<T, Context>(mean.count(), eps, tVar_data);
    math::Pow<T, Context>(mean.count(), 0.5, tVar_data, tVar_data);

    //  divide by var
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, nbychans, spatial_dim, 1, 
                                                                            1.0, 
                                                           tVar_data, SMul_data, 
                                                                            0.0, 
                                                                      Std_data);
    math::Div<T, Context>(output(0)->count(), Ydata, Std_data, Ydata);

    //  release buffer
    ws()->ReleaseBuffer(stddev);
}

template <class Context>
void InstanceNormOp<Context>::RunOnDevice() {
    num = input(0).dim(0); channels = input(0).dim(1);
    spatial_dim = input(0).count(2); nbychans = num * channels;
    vector<TIndex> dims({ num, channels });
    var = ws()->CreateTensor("_t_" + anchor() + "_bn_var");
    mean.Reshape(dims); var->Reshape(dims);

    output(0)->ReshapeLike(input(0));

    //  if true, Act/Exp/Pow/Norm Ops can not exist before when train
    if (inplace) output(0)->Share(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(InstanceNorm);
#ifdef WITH_CUDA
DEPLOY_CUDA(InstanceNorm);
#endif
OPERATOR_SCHEMA(InstanceNorm).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void InstanceNormGradientOp<Context>::RunWithType() {
    INIT_MULTIPLIER(spatial_multiplier, spatial_dim);

    //  get buffer
    stddev = ws()->GetBuffer();
    stddev->ReshapeLike(input(0));

    auto* dYdata = input(-1).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    auto* Std_data = stddev->template mutable_data<T, Context>();
    auto* tVar_data = var->template mutable_data<T, Context>();
    auto* SMul_data = spatial_multiplier->template data<T, Context>();

    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, nbychans, spatial_dim, 1, 
                                                                            1.0, 
                                                           tVar_data, SMul_data, 
                                                                            0.0, 
                                                                      Std_data);

    auto* Ydata = input(-2).template data<T, Context>();
    math::Mul<T, Context>(output(0)->count(), Ydata, dYdata, dXdata);

    //  sum(dE/dY \cdot Y)
    math::Gemv<T, Context>(CblasNoTrans, nbychans, spatial_dim, 
                                                           1.0, 
                                             dXdata, SMul_data, 
                                               0.0, tVar_data);
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, nbychans, spatial_dim, 1, 
                                                                            1.0, 
                                                           tVar_data, SMul_data, 
                                                                            0.0, 
                                                                        dXdata);

    //  sum(dE/dY \cdot Y) \cdot Y
    math::Mul<T, Context>(output(0)->count(), Ydata, dXdata, dXdata);

    //  sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y
    math::Gemv<T, Context>(CblasNoTrans, nbychans, spatial_dim, 
                                                           1.0, 
                                             dYdata, SMul_data, 
                                               0.0, tVar_data);
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, nbychans, spatial_dim, 1, 
                                                                            1.0, 
                                                           tVar_data, SMul_data, 
                                                                            1.0, 
                                                                        dXdata);

    //  dE/dY - mean(dE/dY)- mean(dE/dY \cdot Y) \cdot Y
    //  = dE/dY - mean(sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y)
    math::Axpby<T, Context>(output(0)->count(), 1.0, dYdata,
                                         -1.0 / spatial_dim, 
                                                    dXdata);

    //  divide by var
    math::Div<T, Context>(output(0)->count(), dXdata, Std_data, dXdata);

    //  release buffer
    ws()->ReleaseBuffer(stddev);
}

template <class Context>
void InstanceNormGradientOp<Context>::RunOnDevice() {
    num = input(0).dim(0); channels = input(0).dim(1);
    spatial_dim = input(0).count(2); nbychans = num * channels;
    var = ws()->GetTensor("_t_" + anchor() + "_bn_var");

    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
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