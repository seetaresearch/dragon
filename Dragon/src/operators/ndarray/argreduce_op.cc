#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/ndarray/argreduce_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGUMENTS(X) \
    axis = OperatorBase::Arg<int64_t>("axis", INT_MAX); \
    if (axis != INT_MAX) { \
        axis = axis < 0 ? axis + X.ndim() : axis; \
        CHECK(axis >= 0 && axis < X.ndim()) \
            << "\nExcepted the axis in [-" << X.ndim() << ", " << X.ndim() \
            << "), got " << OperatorBase::Arg<int64_t>("axis", INT_MAX) << "."; \
    }

template <class Context> template <typename T>
void ArgReduceOp<Context>::RunWithType() {
    if (top_k != 1) {
        // It's difficult to implement device code when top_k > 1
        auto* Xdata = Input(0).template data<T, CPUContext>();
        auto* Idata = Output(0)->template mutable_data<int64_t, CPUContext>();
        auto* Vdata = Output(1)->name() != "ignore" ? Output(1)
            ->template mutable_data<T, CPUContext>() : nullptr;
        static CPUContext cctx;
        if (operation == "ARGMAX") {
            kernel::ArgMax<T, CPUContext>(
                outer_dim, inner_dim, axis_dim,
                    top_k, Xdata, Idata, Vdata, &cctx);
        } else if (operation == "ARGMIN") {
            kernel::ArgMin<T, CPUContext>(
                outer_dim, inner_dim, axis_dim,
                    top_k, Xdata, Idata, Vdata, &cctx);
        } else LOG(FATAL) << "Unknown operation: [" << operation << "].";
    } else {
        auto* Xdata = Input(0).template data<T, Context>();
        auto* Idata = Output(0)->template mutable_data<int64_t, Context>();
        auto* Vdata = Output(1)->name() != "ignore" ? Output(1)
            ->template mutable_data<T, Context>() : nullptr;
        if (operation == "ARGMAX") {
            kernel::ArgMax(outer_dim, inner_dim, axis_dim,
                top_k, Xdata, Idata, Vdata, ctx());
        } else if (operation == "ARGMIN") {
            kernel::ArgMin(outer_dim, inner_dim, axis_dim,
                top_k, Xdata, Idata, Vdata, ctx());
        } else LOG(FATAL) << "Unknown operation: [" << operation << "].";
    }
}

template <class Context>
void ArgReduceOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

    if (axis != INT_MAX) {
        axis_dim = Input(0).dim(axis);
        outer_dim = Input(0).count(0, axis);
        inner_dim = Input(0).count(axis + 1);
    } else {
        axis_dim = Input(0).count();
        outer_dim = inner_dim = 1;
    }

    auto dims = Input(0).dims();

    if (!keep_dims) {
        if (axis != INT_MAX) {
            if (top_k > 1) { dims[axis] = top_k; }
            else { dims.erase(dims.begin() + axis); }
        } else {
            if (top_k > 1) dims = { top_k };
            else dims = vector<int64_t>();
        }
    } else {
        if (axis != INT_MAX) { dims[axis] = top_k; }
        else { dims = { top_k }; }
    }

    Output(0)->Reshape(dims); Output(1)->Reshape(dims);

    if (XIsType(Input(0), bool)) RunWithType<bool>();
    else if (XIsType(Input(0), int8_t)) RunWithType<int8_t>();
    else if (XIsType(Input(0), uint8_t)) RunWithType<uint8_t>();
    else if (XIsType(Input(0), int)) RunWithType<int>();
    else if (XIsType(Input(0), int64_t)) RunWithType<int64_t>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0), { 
        "bool", "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
    });
}

DEPLOY_CPU(ArgReduce);
#ifdef WITH_CUDA
DEPLOY_CUDA(ArgReduce);
#endif
OPERATOR_SCHEMA(ArgReduce).NumInputs(1).NumOutputs(2);

NO_GRADIENT(ArgReduce);

#undef DETERMINE_RUNTIME_ARGUMENTS

}  // namespace dragon