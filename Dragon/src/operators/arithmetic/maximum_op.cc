#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/maximum_op.h"

namespace dragon {

template <class Context> template <typename T>
void MaximumOp<Context>::EltwiseRunWithType() {
    auto* X1data = Input(0).template data<T, Context>();
    auto* X2data = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::MaximumE<T, Context>(Output(0)->count(),
        X1data, X2data, Ydata, ctx());
}

template <class Context> template <typename T>
void MaximumOp<Context>::BroadcastRunWithType() {
    T max_val; float x2_val; const T* Xdata; T* Ydata;
    if (Input(0).count() == 1) {
        Output(0)->ReshapeLike(Input(1));
        x2_val = Input(0).template data<float, CPUContext>()[0];
        max_val = dragon_cast<T, float>(x2_val);
        Xdata = Input(1).template data<T, Context>();
        Ydata = Output(0)->template mutable_data<T, Context>();
    } else if (Input(1).count() == 1) {
        Output(0)->ReshapeLike(Input(0));
        x2_val = Input(1).template data<float, CPUContext>()[0];
        max_val = dragon_cast<T, float>(x2_val);
        Xdata = Input(0).template data<T, Context>();
        Ydata = Output(0)->template mutable_data<T, Context>();
    } else { LOG(FATAL) << "Either Input(0) or Input(1) should be a scalar."; }

    kernel::MaximumB<T, Context>(Output(0)->count(),
        Xdata, max_val, Ydata, ctx());
}

template <class Context>
void MaximumOp<Context>::RunOnDevice() {
    if (Input(0).dims() == Input(1).dims()) {
        Output(0)->ReshapeLike(Input(0));
        if (XIsType(Input(0), float)) EltwiseRunWithType<float>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
    } else {
        if (XIsType(Input(0), float)) BroadcastRunWithType<float>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
    }
}

DEPLOY_CPU(Maximum);
#ifdef WITH_CUDA
DEPLOY_CUDA(Maximum);
#endif
OPERATOR_SCHEMA(Maximum).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void MaximumGradientOp<Context>::EltwiseRunWithType() {
    auto* X1data = Input(0).template data<T, Context>();
    auto* X2data = Input(1).template data<T, Context>();
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dX1data = Output(0)->template mutable_data<T, Context>();
    auto* dX2data = Output(1)->template mutable_data<T, Context>();

    kernel::MaximumEGrad<T, Context>(Output(0)->count(),
        X1data, X2data, dYdata, dX1data, dX2data, ctx());
}

template <class Context> template <typename T>
void MaximumGradientOp<Context>::BroadcastRunWithType() {
    T max_val; float x2_val;
    const T* Xdata; T* dX1data; float* dX2data;
    auto* dYdata = Input(-1).template data<T, Context>();
    if (Input(0).count() == 1) {
        x2_val = Input(0).template data<float, CPUContext>()[0];
        max_val = dragon_cast<T, float>(x2_val);
        Xdata = Input(1).template data<T, Context>();
        dX1data = Output(1)->template mutable_data<T, Context>();
        dX2data = Output(0)->template mutable_data<float, Context>();
        kernel::MaximumBGrad<T, Context>(Output(1)->count(),
            Xdata, max_val, dYdata, dX1data, ctx());
    } else if (Input(1).count() == 1) {
        x2_val = Input(1).template data<float, CPUContext>()[0];
        max_val = dragon_cast<T, float>(x2_val);
        Xdata = Input(0).template data<T, Context>();
        dX1data = Output(0)->template mutable_data<T, Context>();
        dX2data = Output(1)->template mutable_data<float, Context>();
        kernel::MaximumBGrad<T, Context>(Output(0)->count(),
            Xdata, max_val, dYdata, dX1data, ctx());
    } else { LOG(FATAL) << "Either Input(0) or Input(1) should be a scalar."; }
    
    // We simply zero the grad of scalar
    math::Set<float, Context>(1, 0, dX2data, ctx());
}

template <class Context>
void MaximumGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));
    Output(1)->ReshapeLike(Input(1));

    if (Input(0).dims() == Input(1).dims()) {
        if (XIsType(Input(0), float)) EltwiseRunWithType<float>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
    } else {
        if (XIsType(Input(0), float)) BroadcastRunWithType<float>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
    }
}

DEPLOY_CPU(MaximumGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(MaximumGradient);
#endif
OPERATOR_SCHEMA(MaximumGradient).NumInputs(3).NumOutputs(2);

class GetMaximumGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetMaximumGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0), GI(1)});
    }
};
REGISTER_GRADIENT(Maximum, GetMaximumGradient);

}   // namespace dragon