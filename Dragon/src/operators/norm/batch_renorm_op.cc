#include "operators/norm/batch_renorm_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/filler.h"

namespace dragon {

template <class Context> template <typename T>
void BatchRenormOp<Context>::RunWithType() {
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
    auto* tDdata = d.template mutable_data<T, Context>();
    auto* tRdata = r->template mutable_data<T, Context>();
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    auto* SMul_data = spatial_multiplier->template data<T, Context>();
    auto* NMul_data = num_multiplier->template data<T, Context>();
    auto* NByC_data = num_by_chans.template mutable_data<T, Context>();
    auto* Std_data = stddev->template mutable_data<T, Context>();
    T* thMean_data = nullptr;
    T* thVar_data = nullptr;
    T* XNorm_data = nullptr;

    const T scale = hFact_data[0] == 0 ? 0 : 1.0 / hFact_data[0];

    if (use_global_stats) {
        math::Scale<T, Context>(mean.count(), scale, hMean_data, tMean_data);
        math::Scale<T, Context>(mean.count(), scale, hVar_data, tVar_data);
    } else {
        thMean_data = t_h_mean.template mutable_data<T, Context>();
        thVar_data = t_h_var.template mutable_data<T, Context>();
        math::Scale<T, Context>(mean.count(), scale, hMean_data, thMean_data);
        math::Scale<T, Context>(mean.count(), scale, hVar_data, thVar_data);
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
        math::Pow<T, Context>(stddev->count(), 2, Ydata, Std_data);
        math::Gemv<T, Context>(CblasNoTrans, nbychans, spatial_dim,
                                         1.0 / (num * spatial_dim), 
                                               Std_data, SMul_data, 
                                                               0.0, 
                                                        NByC_data);
        math::Gemv<T, Context>(CblasTrans, num, channels,
                                                     1.0, 
                                    NByC_data, NMul_data, 
                                                     0.0, 
                                              tVar_data);
        //  update moving average
        hFact_data[0] *= momentum; hFact_data[0] += 1;
        int m = input(0).count() / channels;
        T factor = m > 1 ? T(m) / (m - 1) : 1;
        math::Axpby<T, Context>(mean.count(), 1.0, tMean_data, momentum, hMean_data);
        math::Axpby<T, Context>(mean.count(), factor, tVar_data, momentum, hVar_data);
    }

    //  normalize var
    math::AddScalar<T, Context>(mean.count(), eps, tVar_data);
    math::Pow<T, Context>(mean.count(), 0.5, tVar_data, tVar_data);

    if (!use_global_stats && !is_recomputing) {
        //  normalize history var
        math::AddScalar<T, Context>(mean.count(), eps, thVar_data);
        math::Pow<T, Context>(mean.count(), 0.5, thVar_data, thVar_data);

        //  compute r
        math::Div<T, Context>(mean.count(), tVar_data, thVar_data, tRdata);
        math::Clip<T, Context>(mean.count(), 1.0 / t_r_max, t_r_max, tRdata);

        //  compute d
        math::Sub<T, Context>(mean.count(), tMean_data, thMean_data, tDdata);
        math::Div<T, Context>(mean.count(), tDdata, thVar_data, tDdata);
        math::Clip<T, Context>(mean.count(), -t_d_max, t_d_max, tDdata);

        //  update the bound of r & d
        t_r_max = r_max / (1.0 + (r_max - 1.0) * exp(-t_val));
        t_d_max = d_max / (1.0 + (d_max - 1.0) * exp(-2 * t_val));
        t_val += t_delta;
    }

    //  divide by var
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
    math::Div<T, Context>(stddev->count(), Ydata, Std_data, Ydata);

    if (!use_global_stats) {
        //  store x_norm for backward
        XNorm_data = x_norm->template mutable_data<T, Context>();
        ctx().template Copy<T, Context, Context>(output(0)->count(), XNorm_data, Ydata);

        //  correction: mul by r
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, num, channels, 1,
                                                                        1.0, 
                                                          NMul_data, tRdata,
                                                                        0.0, 
                                                                 NByC_data);
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, nbychans, spatial_dim, 1,
                                                                                1.0, 
                                                               NByC_data, SMul_data, 
                                                                                0.0, 
                                                                          Std_data);
        math::Mul<T, Context>(output(0)->count(), Ydata, Std_data, Ydata);

        //  correction: add by d
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, num, channels, 1,
                                                                        1.0, 
                                                          NMul_data, tDdata, 
                                                                        0.0, 
                                                                 NByC_data);
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, nbychans, spatial_dim, 1,
                                                                                1.0, 
                                                               NByC_data, SMul_data, 
                                                                                1.0, 
                                                                             Ydata);
    }

    //  release buffer
    ws()->ReleaseBuffer(stddev);
}

template <class Context>
void BatchRenormOp<Context>::RunOnDevice() {
    num = input(0).dim(0); channels = input(0).dim(1);
    spatial_dim = input(0).count(2); nbychans = num * channels;
    vector<TIndex> dims(1, channels);
    var = ws()->CreateTensor("_t_" + anchor() + "_bn_var");
    r = ws()->CreateTensor("_t_" + anchor() + "_bn_r");
    mean.Reshape(dims); var->Reshape(dims);
    d.Reshape(dims); r->Reshape(dims);
    t_h_mean.Reshape(dims); t_h_var.Reshape(dims);
    num_by_chans.Reshape(vector<TIndex>(1, nbychans));
    x_norm = ws()->CreateTensor("_t_" + anchor() + "_bn_x_norm");
    x_norm->ReshapeLike(input(0));

    output(0)->ReshapeLike(input(0));

    if (use_stats == -1) use_global_stats = phase() == "TEST" ? true : false;
    else use_global_stats = use_stats == 1 ? true : false;
    is_recomputing = ws()->GetTensor("_t_global_recompute_flag")
                        ->template data<bool, CPUContext>()[0];
    //  if true, Act/Exp/Pow/Norm Ops can not exist before when train
    if (inplace) output(0)->Share(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(BatchRenorm);
#ifdef WITH_CUDA
DEPLOY_CUDA(BatchRenorm);
#endif
OPERATOR_SCHEMA(BatchRenorm).NumInputs(4).NumOutputs(1);

template <class Context> template <typename T>
void BatchRenormGradientOp<Context>::RunWithType() {
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

    if (use_global_stats) {
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
        math::Div<T, Context>(output(0)->count(), dYdata, Std_data, dXdata);
        ws()->ReleaseBuffer(stddev);
        return;
    }

    auto* tRdata = r->template data<T, Context>();
    auto* XNorm_data = x_norm->template data<T, Context>();
    auto* tMean_data = mean.template mutable_data<T, Context>();

    //  buffer <- dE/dY \cdot r
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, num, channels, 1,
                                                                    1.0, 
                                                      NMul_data, tRdata, 
                                                                    0.0, 
                                                             NByC_data);
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, nbychans, spatial_dim, 1,
                                                                            1.0, 
                                                           NByC_data, SMul_data, 
                                                                            0.0, 
                                                                      Std_data);
    math::Mul<T, Context>(output(0)->count(), dYdata, Std_data, Std_data);

    //  sum(dE/dY \cdot Y)
    math::Mul<T, Context>(output(0)->count(), XNorm_data, Std_data, dXdata);
    math::Gemv<T, Context>(CblasNoTrans, nbychans, spatial_dim,
                                                           1.0, 
                                             dXdata, SMul_data, 
                                                           0.0, 
                                                    NByC_data);
    math::Gemv<T, Context>(CblasTrans, num, channels,
                                                 1.0, 
                                NByC_data, NMul_data, 
                                                 0.0, 
                                         tMean_data);
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, num, channels, 1,
                                                                    1.0,
                                                  NMul_data, tMean_data, 
                                                                    0.0, 
                                                             NByC_data);
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, nbychans, spatial_dim, 1,
                                                                            1.0, 
                                                           NByC_data, SMul_data, 
                                                                            0.0, 
                                                                        dXdata);

    //  sum(dE/dY \cdot Y) \cdot Y  
    math::Mul<T, Context>(output(0)->count(), XNorm_data, dXdata, dXdata);

    //  sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y
    math::Gemv<T, Context>(CblasNoTrans, nbychans, spatial_dim,
                                                           1.0, 
                                           Std_data, SMul_data, 
                                                           0.0, 
                                                     NByC_data);
    math::Gemv<T, Context>(CblasTrans, num, channels,
                                                 1.0, 
                                NByC_data, NMul_data, 
                                                 0.0, 
                                         tMean_data);
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, num, channels, 1,
                                                                    1.0,
                                                  NMul_data, tMean_data, 
                                                                    0.0, 
                                                             NByC_data);
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, nbychans, spatial_dim, 1,
                                                                            1.0, 
                                                           NByC_data, SMul_data, 
                                                                   1.0, dXdata);

    //  dE/dY - mean(dE/dY)- mean(dE/dY \cdot Y) \cdot Y
    //  = dE/dY - mean(sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y)
    math::Axpby<T, Context>(output(0)->count(), 1.0, Std_data,
                                   -1.0 / (num * spatial_dim),
                                                      dXdata);

    //  divide var
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
    math::Div<T, Context>(output(0)->count(), dXdata, Std_data, dXdata);

    //  release buffer
    ws()->ReleaseBuffer(stddev);
    ws()->ReleaseBuffer(x_norm, "Common", true);
}

template <class Context>
void BatchRenormGradientOp<Context>::RunOnDevice() {
    num = input(0).dim(0); channels = input(0).dim(1);
    nbychans = num * channels; spatial_dim = input(0).count(2);
    var = ws()->GetTensor("_t_" + anchor() + "_bn_var");
    r = ws()->GetTensor("_t_" + anchor() + "_bn_r");
    mean.ReshapeLike(*var);
    num_by_chans.Reshape(vector<TIndex>(1, nbychans));

    x_norm = ws()->GetTensor("_t_" + anchor() + "_bn_x_norm");
    output(0)->ReshapeLike(input(0));

    if (use_stats == -1) use_global_stats = phase() == "TEST" ? true : false;
    else use_global_stats = use_stats == 1 ? true : false;

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
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