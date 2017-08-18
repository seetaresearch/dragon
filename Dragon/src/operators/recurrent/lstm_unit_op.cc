#include "operators/recurrent/lstm_unit_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void LSTMUnitOp<Context>::RunWithType() {
    auto* c_1 = input(0).template data<T, Context>();
    auto* pre_gate = input(1).template mutable_data<T, Context>();
    auto* cont = has_cont.empty() ? nullptr : cont_t->template data<T, Context>();
    auto* c = output(0)->template mutable_data<T, Context>();
    auto* h = output(1)->template mutable_data<T, Context>();
    kernel::LSTMUnit<T, Context>(input(1).count(), num, channels,
        c_1, pre_gate, cont, pre_gate, c, h);
}

template <class Context>
void LSTMUnitOp<Context>::RunOnDevice() {
    //  input(0):  ----- c_t_1
    //  input(1):  ----- gate_input
    //  output(0): ----- c_t
    //  output(1): ----- h_t
    num = input(0).dim(0);
    channels = input(0).ndim() == 2 ? input(0).dim(1) : input(0).dim(2);
    if (!has_cont.empty()) {
        cont_t = ws()->GetTensor(has_cont);
        CHECK(cont_t->dims() == vector<TIndex>({ 1, num }));
    }
    output(0)->ReshapeLike(input(0));
    output(1)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(LSTMUnit);
#ifdef WITH_CUDA
DEPLOY_CUDA(LSTMUnit);
#endif
OPERATOR_SCHEMA(LSTMUnit).NumInputs(2, 3).NumOutputs(2);


template <class Context> template <typename T>
void LSTMUnitGradientOp<Context>::RunWithType() {
    auto* c_1 = input(0).template data<T, Context>();
    auto* x_act = input(1).template data<T, Context>();
    auto* c = input(2).template data<T, Context>();
    auto* dc = input(3).template data<T, Context>();
    auto* dh = input(4).template data<T, Context>();
    auto* dc_1 = output(0)->template mutable_data<T, Context>();
    auto* dx = output(1)->template mutable_data<T, Context>();
    kernel::LSTMUnitGrad<T, Context>(input(1).count(), num, channels,
        c_1, x_act, c, dc, dh, dc_1, dx);
}

template <class Context>
void LSTMUnitGradientOp<Context>::RunOnDevice() {
    //  input(0):   ----- c_t_1
    //  input(1):   ----- x_act
    //  input(2):   ----- c_t
    //  input(3):   ----- d(c_t)
    //  input(4):   ----- d(h_t)
    //  output(0):  ----- d(c_t_1)
    //  output(1):  ----- d(gate_input)
    num = input(0).dim(0);
    channels = input(0).ndim() == 2 ? input(0).dim(1) : input(0).dim(2);
    output(0)->ReshapeLike(input(0));
    output(1)->ReshapeLike(input(1));
    if (InputSize() != 5) {
        zeros = ws()->CreateTensor("_t_zeros");
        if (zeros->count() < input(0).count())
            zeros->ReshapeLike(input(0));
    }

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
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