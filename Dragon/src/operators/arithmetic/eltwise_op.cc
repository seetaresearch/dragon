#include "utils/math_functions.h"
#include "operators/arithmetic/eltwise_op.h"

namespace dragon {

template <class Context> template <typename T>
void EltwiseOp<Context>::SumRunImpl() {
    auto nelements = Y(0)->count();
    auto* y = Y(0)->template mutable_data<T, Context>();
    math::Set(nelements, cast::to<T>(0.f), y, ctx());
    for (int i = 0; i < XSize(); ++i) {
        auto* x = X(i).template data<T, Context>();
        math::Axpy(nelements, coef_[i], x, y, ctx());
    }
}

template <class Context> template <typename T>
void EltwiseOp<Context>::ProdRunImpl() {
    auto nelements = Y(0)->count();
    auto* y = Y(0)->template mutable_data<T, Context>();
    // Computet the first two inputs
    math::Mul(
        nelements,
        X(0).template data<T, Context>(),
        X(1).template data<T, Context>(),
        y, ctx()
    );
    // Computet the remains
    for (int i = 2; i < XSize(); i++) {
        auto* x = X(i).template data<T, Context>();
        math::Mul(nelements, y, x, y, ctx());
    }
    // Apply the coeffients
    math::Scale(nelements, alpha_, y, y, ctx());
}

template <class Context> template <typename T>
void EltwiseOp<Context>::RunImpl() {
    for (int i = 1; i < XSize(); i++) {
        CHECK(X(i).dims() == X(0).dims())
            << "\nExcepted Input(" << i << ")'s dims as "
            << X(0).DimString() << ",\nwhile got "
            << X(i).DimString() << ".";
    }

    Y(0)->ReshapeLike(X(0));

    if (operation_ == "SUM") SumRunImpl<T>();
    else if (operation_ == "PROD") ProdRunImpl<T>();
    else LOG(FATAL) << "Unknwon Operation: " << operation_;
}

template <class Context>
void EltwiseOp<Context>::RunOnDevice() {
    if (XIsType(X(0), int8_t)) {
        RunImpl<int8_t>();
    } else if (XIsType(X(0), uint8_t)) {
        RunImpl<uint8_t>();
    } else if (XIsType(X(0), int)) {
        RunImpl<int>();
    } else if (XIsType(X(0), int64_t)) {
        RunImpl<int64_t>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16>();
    } else if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), double)) {
        RunImpl<double>();
    } else {
        LOG(FATAL) << DTypeString(X(0), {
            "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
        });
    }
}

template <class Context> template <typename T>
void EltwiseGradientOp<Context>::SumRunImpl() {
    auto nelements = X(-1).count();
    auto* dy = X(-1).template data<T, Context>();

    for (int i = 0; i < YSize(); i++) {
        if (Y(i)->name() == "NULL") continue;
        auto* dx = Y(i)->template mutable_data<T, Context>();
        // Copy the dY to dX and Apply the coeffients
        math::Scale(nelements, coef_[i], dy, dx, ctx());
    }
}

template <class Context> template <typename T>
void EltwiseGradientOp<Context>::ProdRunImpl() {
    auto nelements = X(-1).count();
    auto* dy = X(-1).template data<T, Context>();

    for (int i = 0; i < YSize(); i++) {
        if (Y(i)->name() == "NULL") continue;
        auto* dx = Y(i)->template mutable_data<T, Context>();
        // Compute the first term of dX
        bool initialized = false;
        for (int j = 0; j < YSize(); j++) {
            if (i == j) continue;
            auto* x = X(j).template data<T, Context>();
            if (!initialized) {
                ctx()->template Copy
                    <T, Context, Context>(
                        nelements, dx, x);
                initialized = true;
            } else {
                math::Mul(nelements, x, dx, dx, ctx());
            }
        }
        // Compute the second term of dX, i.e., dY
        math::Mul(nelements, dy, dx, dx, ctx());
        // Apply the coeffients
        math::Scale(nelements, alpha_, dx, dx, ctx());
    }
}

template <class Context> template <typename T>
void EltwiseGradientOp<Context>::RunImpl() {
    for (int i = 0; i < YSize(); i++) {
        CHECK(X(i).dims() == X(0).dims())
            << "\nExcepted Input(" << i << ")'s dims as "
            << X(0).DimString() << ",\n but got "
            << X(i).DimString() << ".";
        Y(i)->ReshapeLike(X(i));
    }

    if (operation_ == "SUM") SumRunImpl<T>();
    else if (operation_ == "PROD") ProdRunImpl<T>();
    else LOG(FATAL) << "Unknwon Operation: " << operation_;
}

template <class Context>
void EltwiseGradientOp<Context>::RunOnDevice() {
    if (XIsType(X(0), int8_t)) {
        RunImpl<int8_t>();
    } else if (XIsType(X(0), uint8_t)) {
        RunImpl<uint8_t>();
    } else if (XIsType(X(0), int)) {
        RunImpl<int>();
    } else if (XIsType(X(0), int64_t)) {
        RunImpl<int64_t>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16>();
    } else if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), double)) {
        RunImpl<double>();
    } else {
        LOG(FATAL) << DTypeString(X(0), {
            "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
        });
    }
}

DEPLOY_CPU(Eltwise);
#ifdef WITH_CUDA
DEPLOY_CUDA(Eltwise);
#endif

DEPLOY_CPU(EltwiseGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(EltwiseGradient);
#endif

OPERATOR_SCHEMA(Eltwise)
     /* X(0), X(1), ... */
    .NumInputs(2, INT_MAX)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(EltwiseGradient)
     /* X(0), X(1), ..., dY */
    .NumInputs(3, INT_MAX)
     /* dX(0), dX(1), ... */
    .NumOutputs(2, INT_MAX);

REGISTER_GRADIENT(Eltwise, SimpleGradientMaker);

}  // namespace dragon