#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/recurrent/lstm_cell_op.h"

namespace dragon {

template <class Context> template <typename T>
void LSTMCellOp<Context>::RunImpl() {
    auto* x = X(0).template mutable_data<T, Context>();
    auto* hx = X(1).template data<T, Context>();
    auto* h = Y(0)->template mutable_data<T, Context>();
    auto* c = Y(1)->template mutable_data<T, Context>();

    kernel::LSTMCell(
        X(1).dim(0),
        X(1).dim(-1),
        hx, x,
        c, h, ctx()
    );
}

template <class Context>
void LSTMCellOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(1));
    Y(1)->ReshapeLike(X(1));

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else {
        LOG(FATAL) << DTypeString(
            X(0), { "float32" }
        );
    }
}

template <class Context> template <typename T>
void LSTMCellGradientOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* hx = X(1).template data<T, Context>();
    auto* c = X(2).template data<T, Context>();
    auto* dh = X(3).template data<T, Context>();
    auto* dc = X(4).template mutable_data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();
    auto* dhx = Y(1)->template mutable_data<T, Context>();

    if (X(-1).name() == "NULL") {
        math::Set(
            X(-1).count(),
            cast::to<T>(0.f),
            dc, ctx()
        );
    }

    kernel::LSTMCellGrad(
        X(1).dim(0),
        X(1).dim(-1),
        hx, x, c, dc, dh,
        dhx, dx, ctx()
    );
}

template <class Context>
void LSTMCellGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));
    Y(1)->ReshapeLike(X(1));

    if (X(-1).name() == "NULL") {
        // dC will be ignored if C is not solved
        // We should Zero-Reset the dC
        X(-1).ReshapeLike(X(-2));
    }

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else {
        LOG(FATAL) << DTypeString(
            X(0), { "float32" }
        );
    }
}

DEPLOY_CPU(LSTMCell);
#ifdef WITH_CUDA
DEPLOY_CUDA(LSTMCell);
#endif

DEPLOY_CPU(LSTMCellGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(LSTMCellGradient);
#endif

OPERATOR_SCHEMA(LSTMCell)
     /* X, HX */
    .NumInputs(2, 3)
     /* H, C */
    .NumOutputs(2);

OPERATOR_SCHEMA(LSTMCellGradient)
     /* X, HX, C, dH, dC */
    .NumInputs(5)
     /* dX, dHX */
    .NumOutputs(2);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override{
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), O(1), GO(0), GO(1) }),
            vector<string>({ GI(0), GI(1) })
        );
    }
    // Fill zero for dCNext
    vector<float> DefaultValues() override{
        return { 1.f, 0.f };
    }
};

}  // namespace

REGISTER_GRADIENT(LSTMCell, GradientMaker);

}  // namespace dragon