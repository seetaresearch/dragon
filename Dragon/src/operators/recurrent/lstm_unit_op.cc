#include "operators/recurrent/lstm_unit_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void LSTMUnitOp<Context>::RunWithType() {
    auto* c_1 = Input(0).template data<T, Context>();
    auto* pre_gate = Input(1).template mutable_data<T, Context>();
    auto* cont = has_cont.empty() ? nullptr : cont_t->template data<T, Context>();
    auto* c = Output(0)->template mutable_data<T, Context>();
    auto* h = Output(1)->template mutable_data<T, Context>();
    kernel::LSTMUnit<T, Context>(Input(1).count(), num, channels,
        c_1, pre_gate, cont, pre_gate, c, h);
}

template <class Context>
void LSTMUnitOp<Context>::RunOnDevice() {
    //  Input(0):  ----- c_t_1
    //  Input(1):  ----- gate_input
    //  Output(0): ----- c_t
    //  Output(1): ----- h_t
    num = Input(0).dim(0);
    channels = Input(0).ndim() == 2 ? Input(0).dim(1) : Input(0).dim(2);
    if (!has_cont.empty()) {
        cont_t = ws()->GetTensor(has_cont);
        CHECK(cont_t->dims() == vector<TIndex>({ 1, num }));
    }
    Output(0)->ReshapeLike(Input(0));
    Output(1)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(LSTMUnit);
#ifdef WITH_CUDA
DEPLOY_CUDA(LSTMUnit);
#endif
OPERATOR_SCHEMA(LSTMUnit).NumInputs(2, 3).NumOutputs(2);


template <class Context> template <typename T>
void LSTMUnitGradientOp<Context>::RunWithType() {
    auto* c_1 = Input(0).template data<T, Context>();
    auto* x_act = Input(1).template data<T, Context>();
    auto* c = Input(2).template data<T, Context>();
    auto* dc = Input(3).template data<T, Context>();
    auto* dh = Input(4).template data<T, Context>();
    auto* dc_1 = Output(0)->template mutable_data<T, Context>();
    auto* dx = Output(1)->template mutable_data<T, Context>();
    kernel::LSTMUnitGrad<T, Context>(Input(1).count(), num, channels,
        c_1, x_act, c, dc, dh, dc_1, dx);
}

template <class Context>
void LSTMUnitGradientOp<Context>::RunOnDevice() {
    //  Input(0):   ----- c_t_1
    //  Input(1):   ----- x_act
    //  Input(2):   ----- c_t
    //  Input(3):   ----- d(c_t)
    //  Input(4):   ----- d(h_t)
    //  Output(0):  ----- d(c_t_1)
    //  Output(1):  ----- d(gate_input)
    num = Input(0).dim(0);
    channels = Input(0).ndim() == 2 ? Input(0).dim(1) : Input(0).dim(2);
    Output(0)->ReshapeLike(Input(0));
    Output(1)->ReshapeLike(Input(1));
    if (InputSize() != 5) {
        zeros = ws()->CreateTensor("/share/zeros");
        if (zeros->count() < Input(0).count())
            zeros->ReshapeLike(Input(0));
    }

    if (Input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(LSTMUnitGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(LSTMUnitGradient);
#endif
OPERATOR_SCHEMA(LSTMUnitGradient).NumInputs(5).NumOutputs(2);

class GetLSTMUnitGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetLSTMUnitGradient);
    vector<OperatorDef> MakeDefs() override{
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), O(0), GO(0), GO(1)},
            vector<string> {GI(0), GI(1)});
    }
    //  fill zero for dc_{T+1}
    vector<float> DefaultValues() override{ return{ 0.0, 1.0 }; }
};
REGISTER_GRADIENT(LSTMUnit, GetLSTMUnitGradient);

}    // namespace dragon