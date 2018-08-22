#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/recurrent/lstm_cell_op.h"

namespace dragon {

template <class Context> template <typename T>
void LSTMCellOp<Context>::RunWithType() {
    auto* XAdata = Input(0).template mutable_data<T, Context>();
    auto* CXdata = Input(1).template data<T, Context>();
    auto* Hdata = Output(0)->template mutable_data<T, Context>();
    auto* Cdata = Output(1)->template mutable_data<T, Context>();

    kernel::LSTMCell<T, Context>(Input(1).count(), Input(1).dim(0),
        Input(1).ndim() == 2 ? Input(1).dim(1) : Input(1).dim(2),
            CXdata, XAdata, Cdata, Hdata, ctx());
}

template <class Context>
void LSTMCellOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(1));
    Output(1)->ReshapeLike(Input(1));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(LSTMCell);
#ifdef WITH_CUDA
DEPLOY_CUDA(LSTMCell);
#endif
OPERATOR_SCHEMA(LSTMCell).NumInputs(2, 3).NumOutputs(2);

template <class Context> template <typename T>
void LSTMCellGradientOp<Context>::RunWithType() {
    auto* XAdata = Input(0).template data<T, Context>();
    auto* CXdata = Input(1).template data<T, Context>();
    auto* Cdata = Input(2).template data<T, Context>();
    auto* dHdata = Input(3).template data<T, Context>();
    auto* dCdata = Input(4).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* dCXdata = Output(1)->template mutable_data<T, Context>();

    kernel::LSTMCellGrad<T, Context>(Input(1).count(), Input(1).dim(0),
        Input(1).ndim() == 2 ? Input(1).dim(1) : Input(1).dim(2),
            CXdata, XAdata, Cdata, dCdata, dHdata, dCXdata, dXdata, ctx());
}

template <class Context>
void LSTMCellGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));
    Output(1)->ReshapeLike(Input(1));

    if (Input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(LSTMCellGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(LSTMCellGradient);
#endif
OPERATOR_SCHEMA(LSTMCellGradient).NumInputs(5).NumOutputs(2);

class GetLSTMCellGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetLSTMCellGradient);
    vector<OperatorDef> MakeDefs() override{
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), O(0), GO(0), GO(1)},
            vector<string> {GI(0), GI(1)});
    }
    //  fill zero for dc_{T+1}
    vector<float> DefaultValues() override{ return{ 0.0, 1.0 }; }
};
REGISTER_GRADIENT(LSTMCell, GetLSTMCellGradient);

}    // namespace dragon