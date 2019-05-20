#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_utils.h"
#include "utils/math_functions.h"
#include "operators/array/where_op.h"

namespace dragon {

template <class Context> template <typename T>
void WhereOp<Context>::RunImpl() {
    const T *a = nullptr, *b = nullptr;
    auto* mask = X(2).template raw_data<Context>();

    if (X(0).count() < X(1).count()) {
        int rows, cols;
        Y(0)->ReshapeLike(X(1));
        a = ws()
            ->template data<T, Context>
                ({ X(1).count() })[0];
        b = X(1).template data<T, Context>();
        auto* ra = X(0).template data<T, Context>();
        if (utils::IsRowwiseBroadcast(
                X(0).dims(), X(1).dims(),
                    &rows, &cols)) {
            math::BroadcastSet(
                rows, cols, 0, ra,
                const_cast<T*>(a), ctx()
            );
        } else if (utils::IsColwiseBroadcast(
                X(0).dims(), X(1).dims(),
                    &rows, &cols)) {
            math::BroadcastSet(
                rows, cols, 1, ra,
                const_cast<T*>(a), ctx()
            );
        } else {
            LOG(FATAL)
                << "Could not broadcast "
                << X(0).DimString()
                << " to "
                << X(1).DimString();
        }
    } else if (X(0).count() > X(1).count()) {
        int rows, cols;
        Y(0)->ReshapeLike(X(0));
        b = ws()
            ->template data<T, Context>
                ({ X(0).count() })[0];
        a = X(0).template data<T, Context>();
        auto* rb = X(1).template data<T, Context>();
        if (utils::IsRowwiseBroadcast(
                X(0).dims(), X(1).dims(),
                    &rows, &cols)) {
            math::BroadcastSet(
                rows, cols, 0, rb,
                const_cast<T*>(b), ctx()
            );
        } else if (utils::IsColwiseBroadcast(
                X(0).dims(), X(1).dims(),
                    &rows, &cols)) {
            math::BroadcastSet(
                rows, cols, 1, rb,
                const_cast<T*>(b), ctx()
            );
        } else {
            LOG(FATAL)
                << "Could not broadcast "
                << X(1).DimString()
                << " to "
                << X(0).DimString();
        }
    } else {
        Y(0)->ReshapeLike(X(0));
        a = X(0).template data<T, Context>();
        b = X(1).template data<T, Context>();
    }

    CHECK_EQ(Y(0)->count(), X(2).count())
        << "\nSize of mask and input should be equal.";

    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::Where(
        Y(0)->count(),
        (const uint8_t*)mask,
        a, b, y, ctx()
    );
}

template <class Context>
void WhereOp<Context>::RunOnDevice() {
    CHECK(XIsType(X(2), bool) || XIsType(X(2), uint8_t))
        << "\nExcepted bool or uint8 mask.";

    DispatchHelper<TensorTypes
        <bool, int8_t, uint8_t, int, int64_t,
               float16, float, double>
    >::Call(this, X(0));
}

template <class Context> template <typename T>
void WhereGradientOp<Context>::RunImpl() {
    T *da = nullptr, *db = nullptr;
    auto* dy = X(-1).template data<T, Context>();
    auto* mask = X(2).template raw_data<Context>();

    int rows, cols, type;
    if (utils::IsRowwiseBroadcast(
            X(0).dims(), X(1).dims(),
                &rows, &cols)) {
        type = 0;
    } else if (utils::IsColwiseBroadcast(
            X(0).dims(), X(1).dims(),
                &rows, &cols)) {
        type = 1;
    }
    vec32_t dims = { rows, cols };
    vec32_t axes = { type };

    if (X(0).count() < X(1).count()) {
        da = ws()
            ->template data<T, Context>
                ({ X(1).count() })[0];
        db = Y(1)->template mutable_data<T, Context>();
        auto* ra = Y(0)->template mutable_data<T, Context>();
        kernel::WhereGrad(
            X(-1).count(),
            (const uint8_t*)mask,
            dy, da, db, ctx()
        );
        kernel::ReduceSum(
            2, dims.data(),
            1, axes.data(),
            1.f, da,
            ra, ctx()
        );
    } else if (X(0).count() > X(1).count()) {
        db = ws()
            ->template data<T, Context>
                ({ X(0).count() })[0];
        da = Y(0)->template mutable_data<T, Context>();
        auto* rb = Y(1)->template mutable_data<T, Context>();
        kernel::WhereGrad(
            X(-1).count(),
            (const uint8_t*)mask,
            dy, da, db, ctx()
        );
        kernel::ReduceSum(
            2, dims.data(),
            1, axes.data(),
            1.f, db,
            rb, ctx()
        );
    } else {
        da = Y(0)->template mutable_data<T, Context>();
        db = Y(1)->template mutable_data<T, Context>();
        kernel::WhereGrad(
            Y(0)->count(),
            (const uint8_t*)mask,
            dy, da, db, ctx()
        );
    }
}

template <class Context>
void WhereGradientOp<Context>::RunOnDevice() {
    CHECK_EQ(X(-1).count(), X(2).count())
        << "\nSize of mask and input should be equal.";

    Y(0)->ReshapeLike(X(0));
    Y(1)->ReshapeLike(X(1));

    CHECK(XIsType(X(2), bool) || XIsType(X(2), uint8_t))
        << "\nExcepted bool or uint8 mask.";

    DispatchHelper<TensorTypes
        <int8_t, uint8_t, int, int64_t,
            float16, float, double>
    >::Call(this, X(0));
}

DEPLOY_CPU(Where);
#ifdef WITH_CUDA
DEPLOY_CUDA(Where);
#endif

DEPLOY_CPU(WhereGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(WhereGradient);
#endif

OPERATOR_SCHEMA(Where)
     /* A, B, M */
    .NumInputs(3)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(WhereGradient)
     /* A, B, M, dY */
    .NumInputs(4)
     /* dA, dB */
    .NumOutputs(2);

namespace {

class GradientMaker : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), I(2), GO(0) }),
            vector<string>({ GI(0), GI(1) })
        );
    }
};

}  // namespace

REGISTER_GRADIENT(Where, GradientMaker);

}  // namespace dragon