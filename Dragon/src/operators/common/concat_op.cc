#include "operators/common/concat_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void ConcatOp<Context>::RunWithType() {
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    for (int i = 0; i < nin; i++){
        auto* Xdata = input(i).template data<T, Context>();
        TIndex count = input(i).count();
        x_concat_dim = input(i).dim(axis);
        kernel::Concat<T, Context>(count, 
                                   outer_dim, inner_dim,
                                   x_concat_dim, y_concat_dim, 
                                   concat_offset, 
                                   Xdata, Ydata, 
                                   &ctx());
        concat_offset += x_concat_dim;
    }
}

template <class Context>
void ConcatOp<Context>::RunOnDevice(){
    concat_dims = input(0).dims();
    for (int i = 1; i < nin; i++){
        CHECK_EQ(concat_dims.size(), input(i).ndim())
            << "\nall inputs must have the same ndim.";
        for (int j = 0; j < concat_dims.size(); j++){
            if (j == axis) continue;
            CHECK_EQ(concat_dims[j], input(i).dim(j))
                << "\nall inputs must have the same dims"
                << ", except concat axis.";
        }
        concat_dims[axis] += input(i).dim(axis);
    }
    y_concat_dim = concat_dims[axis];
    outer_dim = input(0).count(0, axis);
    inner_dim = input(0).count(axis + 1);
    concat_offset = 0;
    output(0)->Reshape(concat_dims);
    if (nin == 1) {
        output(0)->Share(input(0));
        return;
    }

    if (input(0).template IsType<float>()) RunWithType<float>();
    else if (input(0).template IsType<float16>()) RunWithType<float16>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(Concat);
#ifdef WITH_CUDA
DEPLOY_CUDA(Concat);
#endif
OPERATOR_SCHEMA(Concat).NumInputs(1, INT_MAX).NumOutputs(1);

template <class Context> template <typename T>
void ConcatGradientOp<Context>::RunWithType() {
    auto* dYdata = input(-1).template data<T, Context>();
    for (int i = 0; i < nin; i++){
        x_concat_dim = input(i).dim(axis);
        if (output(i)->name() != "ignore") {
            auto* dXdata = output(i)->template mutable_data<T, Context>();
            TIndex count = output(i)->count();
            kernel::ConcatGrad<T, Context>(count, 
                                           outer_dim, inner_dim,
                                           x_concat_dim, y_concat_dim, 
                                           concat_offset,
                                           dYdata, dXdata, 
                                           &ctx());
        }
        concat_offset += x_concat_dim;
    }
}

template <class Context>
void ConcatGradientOp<Context>::RunOnDevice(){
    if (input(-1).name() == "ignore") return;
    concat_dims = input(-1).dims();
    y_concat_dim = concat_dims[axis];
    outer_dim = input(0).count(0, axis);
    inner_dim = input(0).count(axis + 1);
    concat_offset = 0;
    for (int i = 0; i < nin; i++) output(i)->ReshapeLike(input(i));
    if (nin == 1) {
        output(0)->Share(input(-1));
        return;
    }

    if (input(0).template IsType<float>()) RunWithType<float>();
    else if (input(0).template IsType<float16>()) RunWithType<float16>();
    else LOG(FATAL) << "unsupported input types.";
}

template <class Context>
void ConcatGradientOp<Context>::ShareBeforeRun() {
    for (int i = 0; i < OutputSize(); i++) {
        if (output(i)->name() != "ignore") {
            Tensor* dX = ws()->GetBuffer();
            if (dX != nullptr) output(i)->Replace(*dX);
            break;
        }
    }
}

template <class Context>
void ConcatGradientOp<Context>::ClearAfterRun() {
    Tensor* dY = &input(-1);
    ws()->ReleaseBuffer(dY);
}

DEPLOY_CPU(ConcatGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ConcatGradient);
#endif
OPERATOR_SCHEMA(ConcatGradient).NumInputs(2, INT_MAX).NumOutputs(1, INT_MAX);

class GetConcatGradient final : public GradientMakerBase {
public:
    GRADIENT_MAKER_CTOR(GetConcatGradient);
    vector<OperatorDef> MakeDefs() override {
        vector<string> inputs, outputs;
        for (int i = 0; i < def.input_size(); i++) {
            inputs.push_back(def.input(i));
            outputs.push_back(GI(i));
        }
        inputs.push_back(GO(0));
        return SingleDef(def.type() + "Gradient", "", inputs, outputs);
    }
};
REGISTER_GRADIENT(Concat, GetConcatGradient);

}    // namespace dragon

