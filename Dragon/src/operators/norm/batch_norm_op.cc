#include "operators/norm/batch_norm_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/filler.h"

namespace dragon {

template <class Context> template <typename T>
void BatchNormOp<Context>::RunWithType() {
    INIT_MULTIPLIER(num_multiplier, num);
    INIT_MULTIPLIER(spatial_multiplier, spatial_dim);
    TENSOR_FILL(input(1), vector<TIndex>(1, channels));    //  history_mean
    TENSOR_FILL(input(2), vector<TIndex>(1, channels));    //  history_var
    TENSOR_FILL(input(3), vector<TIndex>(1, 1));           //  history_factor

    //  get buffer
    stddev = ws()->GetBuffer();
    stddev->ReshapeLike(input(0));

    auto* hMean_data = input(1).template mutable_data<T, Context>();
    auto* hVar_data = input(2).template mutable_data<T, Context>();
    auto* hFact_data = input(3).template mutable_data<T, CPUContext>();
    auto* tMean_data = mean.template mutable_data<T, Context>();
    auto* tVar_data = var->template mutable_data<T, Context>();
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    auto* SMul_data = spatial_multiplier->template data<T, Context>();
    auto* NMul_data = num_multiplier->template data<T, Context>();
    auto* NByC_data = num_by_chans.template mutable_data<T, Context>();
    auto* Std_data = stddev->template mutable_data<T, Context>();

    if (use_global_stats) {
        const float factor = dragon_cast<float, T>(hFact_data[0]);
        const float scale = factor == 0 ? 0 : 1.0 / factor;
        math::Scale<T, Context>(mean.count(), scale, hMean_data, tMean_data);
        math::Scale<T, Context>(mean.count(), scale, hVar_data, tVar_data);
    } else {
        math::Gemv<T, Context>(CblasNoTrans, nbychans, spatial_dim,
                                         1.0 / (num * spatial_dim), 
                                                  Xdata, SMul_data, 
                                                                 0, 
                                                        NByC_data);
        math::Gemv<T, Context>(CblasTrans, num, channels,
                                                     1.0, 
                                    NByC_data, NMul_data, 
                                                       0, 
                                             tMean_data);
    }

    if (!inplace) {
        ctx().template Copy<T, Context, Context>(input(0).count(), Ydata, Xdata);
    }

    //  subtract mean
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, num, channels, 1,
                                                                    1.0,
                                                  NMul_data, tMean_data,
                                                                    0.0,
                                                             NByC_data);
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, nbychans, spatial_dim, 1,
                                                                           -1.0,
                                                           NByC_data, SMul_data,
                                                                            1.0,
                                                                         Ydata);
    
    if (!use_global_stats && !is_recomputing) {
        //  Var(X) = E((X - EX) ^ 2)
        math::Square<T, Context>(output(0)->count(), Ydata, Std_data);
        math::Gemv<T, Context>(CblasNoTrans, nbychans, spatial_dim,
                                         1.0 / (num * spatial_dim),
                                               Std_data, SMul_data, 
                                                    0.0, NByC_data);
        math::Gemv<T, Context>(CblasTrans, num, channels,
                                                     1.0, 
                                    NByC_data, NMul_data, 
                                                     0.0, 
                                              tVar_data);
        //  handle moving average
        float factor = dragon_cast<float, T>(hFact_data[0]);
        factor *= momentum; factor += 1;
        hFact_data[0] = dragon_cast<T, float>(factor);
        int m = input(0).count() / channels;
        float coeff = m > 1 ? float(m) / (m - 1) : 1;
        //  History(X) = Cur(X) + momentum * History(X) 
        math::Axpby<T, Context>(mean.count(), 1.0, tMean_data, momentum, hMean_data);
        math::Axpby<T, Context>(mean.count(), coeff, tVar_data, momentum, hVar_data);
    }

    //  normalize var
    math::AddScalar<T, Context>(mean.count(), eps, tVar_data);
    math::Sqrt<T, Context>(mean.count(), tVar_data, tVar_data);

    //  divide by stddev
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, num, channels, 1,
                                                                    1.0, 
                                                   NMul_data, tVar_data, 
                                                                    0.0, 
                                                             NByC_data);
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, nbychans, spatial_dim, 1,
                                                                            1.0, 
                                                           NByC_data, SMul_data, 
                                                                            0.0, 
                                                                      Std_data);
    math::Div<T, Context>(output(0)->count(), Ydata, Std_data, Ydata);

    //  release buffer
    ws()->ReleaseBuffer(stddev);
}

template <class Context>
void BatchNormOp<Context>::RunOnDevice() {
    num = input(0).dim(0); channels = input(0).dim(1);
    spatial_dim = input(0).count(2); nbychans = num * channels;
    vector<TIndex> dims(1, channels);
    var = ws()->CreateTensor("_t_" + anchor() + "_bn_var");
    mean.Reshape(dims); var->Reshape(dims);
    num_by_chans.Reshape(vector<TIndex>(1, nbychans));

    output(0)->ReshapeLike(input(0));

    if (use_stats == -1) use_global_stats = phase() == "TEST" ? true : false;
    else use_global_stats = use_stats == 1 ? true : false;
    is_recomputing = ws()->GetTensor("_t_global_recompute_flag")
                         ->template data<bool, CPUContext>()[0];
    //  if true, Act/Exp/Pow/Norm Ops can not exist before when train
    if (inplace) output(0)->Share(input(0));


    if (input(0).template IsType<float>()) RunWithType<float>();
    else if (input(0).template IsType<float16>()) RunWithType<float16>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(BatchNorm);
#ifdef WITH_CUDA
DEPLOY_CUDA(BatchNorm);
#endif
OPERATOR_SCHEMA(BatchNorm).NumInputs(4).NumOutputs(1);

template <class Context> template <typename T>
void BatchNormGradientOp<Context>::RunWithType() {
    INIT_MULTIPLIER(num_multiplier, num);
    INIT_MULTIPLIER(spatial_multiplier, spatial_dim);

    //  get buffer
    stddev = ws()->GetBuffer();
    stddev->ReshapeLike(input(0));

    auto* dYdata = input(-1).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    auto* Std_data = stddev->template mutable_data<T, Context>();
    auto* tVar_data = var->template mutable_data<T, Context>();
    auto* SMul_data = spatial_multiplier->template data<T, Context>();
    auto* NMul_data = num_multiplier->template data<T, Context>();
    auto* NByC_data = num_by_chans.template mutable_data<T, Context>();

    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, num, channels, 1,
                                                                    1.0, 
                                                   NMul_data, tVar_data, 
                                                                    0.0, 
                                                             NByC_data);
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, nbychans, spatial_dim, 1,
                                                                            1.0, 
                                                           NByC_data, SMul_data, 
                                                                            0.0, 
                                                                      Std_data);

    if (use_global_stats) {
        math::Div<T, Context>(output(0)->count(), dYdata, Std_data, dXdata);
        ws()->ReleaseBuffer(stddev);
        return;
    }

    auto* Ydata = input(-2).template data<T, Context>();
    math::Mul<T, Context>(output(0)->count(), Ydata, dYdata, dXdata);

    //  sum(dE/dY \cdot Y)
    math::Gemv<T, Context>(CblasNoTrans, nbychans, spatial_dim,
                                                           1.0, 
                                             dXdata, SMul_data, 
                                                           0.0, 
                                                    NByC_data);
    math::Gemv<T, Context>(CblasTrans, num, channels,
                                                 1.0, 
                                NByC_data, NMul_data, 
                                                 0.0, 
                                          tVar_data);
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, num, channels, 1,
                                                                    1.0, 
                                                   NMul_data, tVar_data, 
                                                                    0.0, 
                                                             NByC_data);
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, nbychans, spatial_dim, 1,
                                                                            1.0, 
                                                           NByC_data, SMul_data, 
                                                                            0.0, 
                                                                        dXdata);

    //  sum(dE/dY \cdot Y) \cdot Y  
    math::Mul<T, Context>(output(0)->count(), Ydata, dXdata, dXdata);

    //  sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y
    math::Gemv<T, Context>(CblasNoTrans, nbychans, spatial_dim,
                                                           1.0, 
                                             dYdata, SMul_data, 
                                                           0.0, 
                                                    NByC_data);
    math::Gemv<T, Context>(CblasTrans, num, channels,
                                                 1.0, 
                                NByC_data, NMul_data, 
                                                 0.0, 
                                          tVar_data);
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, num, channels, 1,
                                                                    1.0, 
                                                   NMul_data, tVar_data, 
                                                                    0.0, 
                                                             NByC_data);
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, nbychans, spatial_dim, 1,
                                                                            1.0, 
                                                           NByC_data, SMul_data, 
                                                                            1.0, 
                                                                        dXdata);

    //   dE/dY - mean(dE/dY)- mean(dE/dY \cdot Y) \cdot Y
    // = dE/dY - mean(sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y)
    math::Axpby<T, Context>(output(0)->count(), 1.0, dYdata,
                                 -1.0 / (num * spatial_dim), 
                                                    dXdata);

    //  divide by stddev
    math::Div<T, Context>(output(0)->count(), dXdata, Std_data, dXdata);

    //  release buffer
    ws()->ReleaseBuffer(stddev);
}

template <class Context>
void BatchNormGradientOp<Context>::RunOnDevice() {
    num = input(0).dim(0); channels = input(0).dim(1);
    spatial_dim = input(0).count(2); nbychans = num * channels;
    var = ws()->GetTensor("_t_" + anchor() + "_bn_var");
    num_by_chans.Reshape(vector<TIndex>(1, nbychans));

    output(0)->ReshapeLike(input(0));

    if (use_stats == -1) use_global_stats = phase() == "TEST" ? true : false;
    else use_global_stats = use_stats == 1 ? true : false;
    
    if (input(0).template IsType<float>()) RunWithType<float>();
    else if (input(0).template IsType<float16>()) RunWithType<float16>();
    else LOG(FATAL) << "unsupported input types.";
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