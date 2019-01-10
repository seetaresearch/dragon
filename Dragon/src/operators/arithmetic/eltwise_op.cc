#include "utils/math_functions.h"
#include "operators/arithmetic/eltwise_op.h"

namespace dragon {

template <class Context> template <typename T>
void EltwiseOp<Context>::SumRunWithType() {
    auto nelements = Output(0)->count();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    math::Set(nelements, cast::to<T>(0.f), Ydata, ctx());
    for (int i = 0; i < InputSize(); ++i) {
        math::Axpy(nelements, coeffs[i],
            Input(i).template data<T, Context>(), Ydata, ctx());
    }
}

template <class Context> template <typename T>
void EltwiseOp<Context>::ProdRunWithType() {
    auto nelements = Output(0)->count();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    // Computet the first two inputs
    math::Mul(nelements,
        Input(0).template data<T, Context>(),
            Input(1).template data<T, Context>(),
                Ydata, ctx());
    // Computet the remains
    for (int i = 2; i < InputSize(); i++) {
        auto* Xdata = Input(i).template data<T, Context>();
        math::Mul(nelements, Ydata, Xdata, Ydata, ctx());
    }
    // Apply the coeffients
    math::Scale(nelements, alpha, Ydata, Ydata, ctx());
}

template <class Context> template <typename T>
void EltwiseOp<Context>::RunWithType() {
    for (int i = 1; i < InputSize(); i++) {
        CHECK(Input(i).dims() == Input(0).dims())
            << "\nExcepted Input(" << i << ")'s dims as "
            << Input(0).DimString() << ",\n but got "
            << Input(i).DimString() << ".";
    } Output(0)->ReshapeLike(Input(0));

    if (operation == "SUM") SumRunWithType<T>();
    else if (operation == "PROD") ProdRunWithType<T>();
    else LOG(FATAL) << "Unknwon operation: " << operation;
}

template <class Context>
void EltwiseOp<Context>::RunOnDevice() {
    if (XIsType(Input(0), int8_t)) RunWithType<int8_t>();
    else if (XIsType(Input(0), uint8_t)) RunWithType<uint8_t>();
    else if (XIsType(Input(0), int)) RunWithType<int>();
    else if (XIsType(Input(0), int64_t)) RunWithType<int64_t>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0), {
        "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
    });
}

DEPLOY_CPU(Eltwise);
#ifdef WITH_CUDA
DEPLOY_CUDA(Eltwise);
#endif
OPERATOR_SCHEMA(Eltwise).NumInputs(2, INT_MAX).NumOutputs(1);

template <class Context> template <typename T>
void EltwiseGradientOp<Context>::SumRunWithType() {
    auto nelements = Input(-1).count();
    auto* dYdata = Input(-1).template data<T, Context>();

    for (int i = 0; i < OutputSize(); i++) {
        if (Output(i)->name() == "ignore") continue;
        auto* dXdata = Output(i)->template mutable_data<T, Context>();
        // Copy the dY to dX and Apply the coeffients
        math::Scale(nelements, coeffs[i], dYdata, dXdata, ctx());
    }
}

template <class Context> template <typename T>
void EltwiseGradientOp<Context>::ProdRunWithType() {
    auto nelements = Input(-1).count();
    auto* dYdata = Input(-1).template data<T, Context>();

    for (int i = 0; i < OutputSize(); i++) {
        if (Output(i)->name() == "ignore") continue;
        auto* dXdata = Output(i)->template mutable_data<T, Context>();
        // Compute the first term of dX
        bool initialized = false;
        for (int j = 0; j < OutputSize(); j++) {
            if (i == j) continue;
            auto* Xdata = Input(j).template data<T, Context>();
            if (!initialized) {
                ctx()->template Copy<T, Context, Context>(
                    nelements, dXdata, Xdata);
                initialized = true;
            } else {
                math::Mul(nelements, Xdata, dXdata, dXdata, ctx());
            }
        }
        // Compute the second term of dX, i.e., dY
        math::Mul(nelements, dYdata, dXdata, dXdata, ctx());
        // Apply the coeffients
        math::Scale(nelements, alpha, dXdata, dXdata, ctx());
    }
}

template <class Context> template <typename T>
void EltwiseGradientOp<Context>::RunWithType() {
    for (int i = 0; i < OutputSize(); i++) {
        CHECK(Input(i).dims() == Input(0).dims())
            << "\nExcepted Input(" << i << ")'s dims as "
            << Input(0).DimString() << ",\n but got "
            << Input(i).DimString() << ".";
        Output(i)->ReshapeLike(Input(i));
    }

    if (operation == "SUM") SumRunWithType<T>();
    else if (operation == "PROD") ProdRunWithType<T>();
    else LOG(FATAL) << "Unknwon operation: " << operation;
}

template <class Context>
void EltwiseGradientOp<Context>::RunOnDevice() {
    if (XIsType(Input(0), int8_t)) RunWithType<int8_t>();
    else if (XIsType(Input(0), uint8_t)) RunWithType<uint8_t>();
    else if (XIsType(Input(0), int)) RunWithType<int>();
    else if (XIsType(Input(0), int64_t)) RunWithType<int64_t>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0), {
        "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
    });
}

DEPLOY_CPU(EltwiseGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(EltwiseGradient);
#endif

OPERATOR_SCHEMA(EltwiseGradient)
    .NumInputs(3, INT_MAX)
    .NumOutputs(2, INT_MAX);

REGISTER_GRADIENT(Eltwise, SimpleGradientMaker);

}  // namespace dragon