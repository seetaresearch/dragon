#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_utils.h"
#include "utils/math_functions.h"
#include "operators/array/masked_select_op.h"

namespace dragon {

template <class Context> template <typename T>
void MaskedSelectOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* mask = X(1).template raw_data<Context>();

    auto* scratch = ws()->CreateTensor("/share/data");
    auto* indices = ws()->CreateTensor(unique_name("indices"));

    kernel::MaskedSelect(
        X(0).count(),
        (const uint8_t*)mask, x,
        indices, scratch,
        Y(0), ctx()
    );
}

template <class Context>
void MaskedSelectOp<Context>::RunOnDevice() {
    CHECK_EQ(X(0).count(), X(1).count())
        << "\nSize of mask and input should be equal.";

    CHECK(XIsType(X(1), bool) || XIsType(X(1), uint8_t))
        << "\nExcepted bool or uint8 mask.";

    DispatchHelper<TensorTypes
        <bool, int8_t, uint8_t, int, int64_t,
               float16, float, double>
    >::Call(this, X(0));
}

template <class Context> template <typename T>
void MaskedSelectGradientOp<Context>::RunImpl() {
    auto* dy = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    auto* i = ws()
        ->GetTensor(unique_name("indices"))
        ->template data<int64_t, Context>();

    kernel::MaskedSelectGrad(
        X(0).count(),
        X(1).count(),
        i, dy,
        dx, ctx()
    );
}

template <class Context>
void MaskedSelectGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <bool, int8_t, uint8_t, int, int64_t,
               float16, float, double>
    >::Call(this, X(0));
}

DEPLOY_CPU(MaskedSelect);
#ifdef WITH_CUDA
DEPLOY_CUDA(MaskedSelect);
#endif

DEPLOY_CPU(MaskedSelectGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(MaskedSelectGradient);
#endif

OPERATOR_SCHEMA(MaskedSelect)
     /* X, M */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(MaskedSelectGradient)
     /* X, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), GO(0) }),
            vector<string>({ GI(0)} )
        );
    }
};

}  // namespace

REGISTER_GRADIENT(MaskedSelect, GradientMaker);

}  // namespace dragon