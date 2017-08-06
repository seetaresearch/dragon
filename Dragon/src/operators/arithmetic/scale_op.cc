#include "operators/arithmetic/scale_op.h"
#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void ScaleOp<Context>::RunWithType() {
    CHECK_LT(axis, (int)input(0).ndim());
    const vector<TIndex>::const_iterator& dim_start =
        input(0).dims().begin() + axis;
    if (num_axes == -1) num_axes = (int)input(0).ndim() - axis;
    CHECK_LE(axis + num_axes, (int)input(0).ndim());
    const vector<TIndex>::const_iterator& dim_end = dim_start + num_axes;
    vector<TIndex> param_dims(dim_start, dim_end);
    TENSOR_FILL(input(1), param_dims);
    if (InputSize() > 2) {
        TENSOR_FILL(input(2), param_dims);
        inner_dim = input(0).count(axis + num_axes);
        INIT_MULTIPLIER(bias_multiplier, inner_dim);
    }

    if (InputSize() > 2) {
        kernel::Scale<T, Context>(axis, &input(0), &input(1),
                                  &input(2), bias_multiplier, 
                                                  output(0));
    } else {
        kernel::Scale<T, Context>(axis, &input(0), &input(1),
                                            nullptr, nullptr, 
                                                  output(0));
    }
}

template <class Context>
void ScaleOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else if (input(0).template IsType<float16>()) RunWithType<float16>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(Scale);
#ifdef WITH_CUDA
DEPLOY_CUDA(Scale);
#endif
OPERATOR_SCHEMA(Scale).NumInputs(2, 3).NumOutputs(1);

template <class Context> template <typename T>
void ScaleGradientOp<Context>::BiasRunWithType() {
    output(2)->ReshapeLike(input(1));
    INIT_MULTIPLIER(bias_multiplier, inner_dim);
    auto* BMul_data = this->bias_multiplier->template data<T, Context>();
    auto* dYdata = input(-1).template data<T, Context>();
    auto* dBias = output(2)->template mutable_data<T, Context>();

    for (int n = 0; n < outer_dim; n++) {
        math::Gemv<T, Context>(CblasNoTrans, scale_dim, inner_dim,
                                                              1.0, 
                                                dYdata, BMul_data, 
                                                              1.0, 
                                                           dBias);
        dYdata += dim;
    }
}

template <class Context> template <typename T>
void ScaleGradientOp<Context>::ScaleRunWithType() {
    output(0)->ReshapeLike(input(0));
    output(1)->ReshapeLike(input(1));
    INIT_MULTIPLIER(sum_multiplier, sum_dim);
    sum_result.Reshape(vector<TIndex>(1, outer_dim * scale_dim));
    bool is_eltwise = (input(0).count() == input(1).count());
    auto* dYdata = input(-1).template data<T, Context>();
    auto* Xdata = input(0).template data<T, Context>();
    auto* dScale = output(1)->template mutable_data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    auto* tmp_data = (is_eltwise ? dScale : dXdata);
    auto* SMul_data = sum_multiplier->template mutable_data<T, Context>();

    math::Mul<T, Context>(output(0)->count(), dYdata, Xdata, tmp_data);

    if (!is_eltwise) {
        T* SRes_data = nullptr;
        if (inner_dim == 1) {
            SRes_data = tmp_data;
        } else if (sum_result.count() == 1) {    //  handle inner only
            dScale = output(1)->template mutable_data<T, CPUContext>();
            T result = math::Dot<T, Context>(inner_dim, tmp_data, SMul_data);
            *dScale += result;
        } else {
            SRes_data = (outer_dim == 1) ?  //  handle scale only
                dScale : sum_result.template mutable_data<T, Context>();
            math::Gemv<T, Context>(CblasNoTrans, sum_result.count(), inner_dim,
                                                                           1.0, 
                                                           tmp_data, SMul_data, 
                                               SRes_data == dScale ? 1.0 : 0.0, 
                                                                    SRes_data);
        } 
        if (outer_dim != 1) {
            if (scale_dim == 1) {    //  handle outer only
                T result = math::Dot<T, Context>(outer_dim, SMul_data, SRes_data);
                *dScale += result;
            } else {
                math::Gemv<T, Context>(CblasTrans, outer_dim, scale_dim,
                                                                    1.0, 
                                                   SRes_data, SMul_data, 
                                                                    1.0, 
                                                                dScale);
            }
        }
    }
}

template <class Context> template <typename T>
void ScaleGradientOp<Context>::RunWithType() {
    output(0)->ReshapeLike(input(0));
    kernel::ScaleGrad<float, Context>(axis, &input(-1), &input(1), output(0));
}

template <class Context>
void ScaleGradientOp<Context>::RunOnDevice() {
    if (num_axes == -1) num_axes = (int)input(0).ndim() - axis;
    outer_dim = input(0).count(0, axis);
    inner_dim = input(0).count(axis + num_axes);
    scale_dim = input(1).count();
    sum_dim = std::max(outer_dim, inner_dim);
    dim = scale_dim * inner_dim;

    if (input(0).template IsType<float>()) {
        if (output(2)->name() != "ignore") BiasRunWithType<float>();
        if (output(1)->name() != "ignore") ScaleRunWithType<float>();
        if (output(0)->name() != "ignore") RunWithType<float>();  
    } else {
        LOG(FATAL) << "unsupported input types.";
    }
}

template <class Context>
void ScaleGradientOp<Context>::ShareBeforeRun() {
    Tensor* dX = ws()->GetBuffer();
    if (dX != nullptr) output(0)->Replace(*dX);
}

template <class Context>
void ScaleGradientOp<Context>::ClearAfterRun() {
    Tensor* dY = &input(-1);
    ws()->ReleaseBuffer(dY);
}

DEPLOY_CPU(ScaleGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ScaleGradient);
#endif
OPERATOR_SCHEMA(ScaleGradient).NumInputs(3).NumOutputs(3);

class GetScaleGradient final : public GradientMakerBase {
public:
    GRADIENT_MAKER_CTOR(GetScaleGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0), GI(1), GI(2)});
    }
};
REGISTER_GRADIENT(Scale, GetScaleGradient);

}    // namespace dragon