#include "operators/ndarray/reduce_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void ReduceOp<Context>::SumRunWithType() {
    auto* Xdata = input(0).template data<T, Context>();
    if (axis == -1) {
        INIT_MULTIPLIER(multiplier, input(0).count());
        auto* Ydata = output(0)->template mutable_data<T, CPUContext>();
        auto* Mdata = multiplier->template data<T, Context>();
        Ydata[0] = math::Dot<T, Context>(input(0).count(), Mdata, Xdata);
    } else {
        auto* Ydata = output(0)->template mutable_data<T, Context>();
        kernel::Sum<T, Context>(count, axis_dim, inner_dim, Xdata, Ydata);
    }
}

template <class Context> template <typename T>
void ReduceOp<Context>::MeanRunWithType() {
    SumRunWithType<T>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    T coeff = axis != -1 ? 1.0 / axis_dim : 1.0 / input(0).count();
    math::Scal<T, Context>(output(0)->count(), coeff, Ydata);
}

template <class Context>
void ReduceOp<Context>::RunOnDevice() {
    if (axis != -1) {
        count = input(0).count() / input(0).dim(axis);
        axis_dim = input(0).dim(axis);
    }
    inner_dim = input(0).count(axis + 1);
    vector<TIndex> dims = input(0).dims();
    if (!keep_dims) {
        if (axis != -1) dims.erase(dims.begin() + axis);
        else dims = vector<TIndex>(1, 1);
    } else {
        if (axis != -1) dims[axis] = 1;
        else dims = vector<TIndex>(input(0).ndim(), 1);
    }
    output(0)->Reshape(dims);

    if (operation == "SUM") {
        if (input(0).template IsType<float>()) SumRunWithType<float>();
        else LOG(FATAL) << "Unsupported input types.";
    } 
    else if (operation == "MEAN") {
        if (input(0).template IsType<float>()) MeanRunWithType<float>();
        else LOG(FATAL) << "Unsupported input types.";
    } 
    else {
        LOG(FATAL) << "Unknown operation: [" << operation << "].";
    }
}

DEPLOY_CPU(Reduce);
#ifdef WITH_CUDA
DEPLOY_CUDA(Reduce);
#endif
OPERATOR_SCHEMA(Reduce).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void ReduceGradientOp<Context>::SumRunWithType() {
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    if (axis == -1) {
        auto* dYdata = input(-1).template data<T, CPUContext>();
        math::Set<T, Context>(output(0)->count(), dYdata[0], dXdata);
    } else {
        auto* dYdata = input(-1).template data<T, Context>();
        kernel::SumGrad<T, Context>(count, axis_dim, inner_dim, 1.0, dYdata, dXdata);
    }
}

template <class Context> template <typename T>
void ReduceGradientOp<Context>::MeanRunWithType() {
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    if (axis == -1) {
        auto* dYdata = input(-1).template data<T, CPUContext>();
        math::Set<T, Context>(output(0)->count(), dYdata[0] / input(0).count(), dXdata);
    } else {
        auto* dYdata = input(-1).template data<T, Context>();
        kernel::SumGrad<T, Context>(count, axis_dim, inner_dim, 1.0 / axis_dim, dYdata, dXdata);
    }
}

template <class Context>
void ReduceGradientOp<Context>::RunOnDevice() {
    if (axis != -1) {
        count = input(0).count() / input(0).dim(axis);
        axis_dim = input(0).dim(axis);
    }
    inner_dim = input(0).count(axis + 1);
    output(0)->ReshapeLike(input(0));

    if (operation == "SUM") {
        if (input(0).template IsType<float>()) SumRunWithType<float>();
        else LOG(FATAL) << "Unsupported input types.";
    } else if (operation == "MEAN") {
        if (input(0).template IsType<float>()) MeanRunWithType<float>();
        else LOG(FATAL) << "Unsupported input types.";
    } else {
        LOG(FATAL) << "Unknown operation: [" << operation << "].";
    }
}

DEPLOY_CPU(ReduceGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ReduceGradient);
#endif
OPERATOR_SCHEMA(ReduceGradient).NumInputs(2).NumOutputs(1);

class GetReduceGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetReduceGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Reduce, GetReduceGradient);

}    // namespace dragon