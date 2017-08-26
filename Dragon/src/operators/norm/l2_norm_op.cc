#include "operators/norm/l2_norm_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context> template <typename T>
void L2NormOp<Context>::RunWithType() {
    INIT_MULTIPLIER(multiplier, dim);

    //  normalize by outer dim independently
    buffer = ws()->GetBuffer();
    vector<TIndex> dims = input(0).dims();
    for (int i = 0; i < axis; i++) dims[i] = 1;
    buffer->Reshape(dims);

    //  normalize by inner_dim independently if not across it
    norm = ws()->CreateTensor("_t_" + anchor() + "_l2norm_normalizer");
    dims = input(0).dims();
    for (int i = axis; i < end_axis; i++) dims[i] = 1;
    norm->Reshape(dims);

    auto* Xdata = input(0).template data<T, Context>();
    auto* DMuldata = multiplier->data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    auto* Bdata = buffer->template mutable_data<T, Context>();
    auto* Ndata = norm->template mutable_data<T, Context>();

    for (int n = 0; n < outer_dim; n++) {
        if (across_inner) {
            auto* Ndata_ = norm->template mutable_data<float, CPUContext>();
            float sum_of_sqr = math::Dot<T, Context>(buffer->count(), Xdata, Xdata);
            Ndata_[n] = pow(sum_of_sqr + eps, 0.5);
            math::Scale<T, Context>(buffer->count(), 1.0 / Ndata_[n], Xdata, Ydata);
        } else {
            math::Set<T, Context>(norm->count(), dragon_cast<T, float>(eps), Ndata);
            math::Square<T, Context>(buffer->count(), Xdata, Bdata);
            //  compute T1 = \sum_{i} x_{i,j}^{2}
            math::Gemv<T, Context>(CblasTrans, dim, inner_dim, 
                                                          1.0,
                                              Bdata, DMuldata, 
                                                          1.0, 
                                                       Ndata);
            //  compute T2 = \sqrt{T1}
            math::Sqrt<T, Context>(inner_dim, Ndata, Ndata);
            //  compute T3 = x / [(T2)]_{dim} 
            math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, dim, inner_dim, 1,
                                                                             1.0, 
                                                                 DMuldata, Ndata, 
                                                                             0.0, 
                                                                          Bdata);
            math::Div<T, Context>(buffer->count(), Xdata, Bdata, Ydata);
            Ndata += inner_dim;
        }
        Xdata += buffer->count();
        Ydata += buffer->count();
    }

    //  release buffer
    ws()->ReleaseBuffer(buffer);
}

template <class Context>
void L2NormOp<Context>::RunOnDevice() {
    if (num_axes >= 0) {
        if (num_axes == 0) num_axes += 1;
    } else num_axes = (int)input(0).ndim() - axis;
    end_axis = axis + num_axes;
    CHECK_LE(end_axis, int(input(0).ndim()));

    //  do statistics through [axis, end_axis)
    outer_dim = input(0).count(0, axis);
    dim = input(0).count(axis, axis + num_axes);
    inner_dim = input(0).count(axis + num_axes);
    if (inner_dim == 1) across_inner = true;
    else across_inner = false;

    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
#ifdef WITH_CUDA_FP16
    else if (input(0).template IsType<float16>()) RunWithType<float16>();
#endif
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(L2Norm);
#ifdef WITH_CUDA
DEPLOY_CUDA(L2Norm);
#endif
OPERATOR_SCHEMA(L2Norm).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void L2NormGradientOp<Context>::RunWithType() {
    INIT_MULTIPLIER(multiplier, dim);

    //  normalize by inner_dim independently if not across it
    norm = ws()->GetTensor("_t_" + anchor() + "_l2norm_normalizer");
    buffer = ws()->GetBuffer();
    vector<TIndex> dims = input(0).dims();
    for (int i = 0; i < axis; i++) dims[i] = 1;
    buffer->Reshape(dims);
    buffer_inner = ws()->GetBuffer();
    buffer_inner->Reshape(vector<TIndex>(1, inner_dim));

    auto* Xdata = input(0).template data<T, Context>();
    auto* dYdata = input(-1).template data<T, Context>();
    auto* DMuldata = multiplier->data<T, Context>();
    auto* Ndata = norm->template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    auto* Bdata = buffer->template mutable_data<T, Context>();
    auto* BInnerdata = buffer_inner->template mutable_data<T, Context>();

    for (int n = 0; n < outer_dim; n++) {
        if (across_inner) {
            Ndata = norm->template data<T, CPUContext>();
            T sum_of_x_mul_dy = math::Dot<T, Context>(buffer->count(), Xdata, dYdata);
            math::Scale<T, Context>(buffer->count(), sum_of_x_mul_dy / Ndata[n] / Ndata[n], Xdata, dXdata);
            math::Sub<T, Context>(buffer->count(), dYdata, dXdata, dXdata);
            math::Scal<T, Context>(buffer->count(), T(1.0 / Ndata[n]), dXdata);
        } else {
            //  compute \sum_{i} x_{i, j}dy_{i, j}
            math::Mul<T, Context>(buffer->count(), Xdata, dYdata, Bdata);
            math::Gemv<T, Context>(CblasTrans, dim, inner_dim, 
                                                          1.0, 
                                              Bdata, DMuldata, 
                                                          0.0, 
                                                  BInnerdata);
            //  compute T1 = x[(\sum_{i} x_{i, j}dy_{i, j})]_{dim}
            math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, dim, inner_dim, 1, 
                                                                             1.0, 
                                                            DMuldata, BInnerdata, 
                                                                             0.0, 
                                                                          Bdata);
            math::Mul<T, Context>(buffer->count(), Xdata, Bdata, dXdata);
            //  compute T2 = T1 / Normalizer^{2}
            math::Pow<T, Context>(inner_dim, 2.0, Ndata, BInnerdata);
            math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, dim, inner_dim, 1, 
                                                                             1.0, 
                                                            DMuldata, BInnerdata, 
                                                                             0.0, 
                                                                          Bdata);
            math::Div<T, Context>(buffer->count(), dXdata, Bdata, dXdata);
            //  compute T3 = (dy - T2) / Normalizer
            math::Sub<T, Context>(buffer->count(), dYdata, dXdata, dXdata);
            math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, dim, inner_dim, 1, 
                                                                             1.0, 
                                                                 DMuldata, Ndata, 
                                                                             0.0, 
                                                                          Bdata);
            math::Div<T, Context>(buffer->count(), dXdata, Bdata, dXdata);
            Ndata += inner_dim;
        }
        Xdata += buffer->count();
        dYdata += buffer->count();
        dXdata += buffer->count();
    }

    // release buffer
    ws()->ReleaseBuffer(buffer_inner);
    ws()->ReleaseBuffer(buffer);
}

template <class Context>
void L2NormGradientOp<Context>::RunOnDevice() {
    if (num_axes >= 0) {
        if (num_axes == 0) num_axes += 1;
    } else { num_axes = (int)input(0).ndim() - axis; }
    end_axis = axis + num_axes;
    CHECK_LE(end_axis, int(input(0).ndim()));

    //  do statistics through [axis, end_axis)
    outer_dim = input(0).count(0, axis);
    dim = input(0).count(axis, axis + num_axes);
    inner_dim = input(0).count(axis + num_axes);
    if (inner_dim == 1) across_inner = true;
    else across_inner = false;

    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(L2NormGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(L2NormGradient);
#endif
OPERATOR_SCHEMA(L2NormGradient).NumInputs(2).NumOutputs(1);

class GetL2NormGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetL2NormGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(L2Norm, GetL2NormGradient);

}    // namespace dragon