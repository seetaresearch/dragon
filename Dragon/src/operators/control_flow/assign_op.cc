#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_utils.h"
#include "utils/math_functions.h"
#include "operators/control_flow/assign_op.h"

namespace dragon {

#define TENSOR_FROM_VECTOR(tensor, vec, T) \
    { \
        tensor.Reshape({ (int64_t)vec.size() }); \
        auto* data = tensor.template mutable_data<T, CPUContext>(); \
        for (int i = 0; i < vec.size(); i++) data[i] = (T)vec[i]; \
    }

template <class Context> template <typename T>
void AssignOp<Context>::RunImpl() {
    const T* x = nullptr;
    auto* y = Y(0)->template mutable_data<T, Context>();

    if (X(0).count() < X_.count()) {
        int rows, cols;
        auto* scratch = ws()
            ->template data<T, Context>
                ({ X_.count() })[0];
        auto* rx = X(0).template data<T, Context>();
        if (utils::IsRowwiseBroadcast(
                X_.dims(), X(0).dims(),
                    &rows, &cols)) {
            math::BroadcastSet(
                rows, cols, 0,
                rx, scratch, ctx()
            );
        } else if (utils::IsColwiseBroadcast(
                X_.dims(), X(0).dims(),
                    &rows, &cols)) {
            math::BroadcastSet(
                rows, cols, 1,
                rx, scratch, ctx()
            );
        } else {
            LOG(FATAL)
                << "Could not broadcast "
                << X(0).DimString()
                << " to "
                << X_.DimString();
        }
        x = scratch;
    } else if (X(0).count() == X_.count()) {
        x = X(0).template data<T, Context>();
    } else {
        LOG(FATAL)
            << "Could not assign "
            << X(0).DimString()
            << " to "
            << Y(0)->DimString();
    }

    // Apply a simple Nd-Broadcast solution
    kernel::Assign(
        X_.count(),
        X_dims_.count(),
        X_dims_.template data<int, Context>(),
        Y_strides_.template data<int, Context>(),
        X_starts_.template data<int, Context>(),
        x, y, ctx()
    );
}

template <class Context>
void AssignOp<Context>::Setup() {
    st_.assign((size_t)Y(0)->ndim(), 0);
    ed_.assign(st_.size(), 0);

    // Determine the starts
    int nstarts = GET_ARGS_SIZE(starts);
    for (int i = 0; i < st_.size(); i++)
        if (i < nstarts) st_[i] = starts(i);
 
    // Determine the ends
    int nsizes = GET_ARGS_SIZE(sizes);
    for (int i = 0; i < ed_.size(); i++) {
        ed_[i] = Y(0)->dim(i);
        if (i < nsizes) {
            auto len = sizes(i);
            if (len > 0) { ed_[i] = st_[i] + len; }
            else if (len == 0) { ed_[i] = st_[i] + 1; }
        }
    }

    // Check starts and ends
    for (int i = 0; i < st_.size(); i++) {
        CHECK(st_[i] >= 0 && st_[i] < Y(0)->dim(i))
            << "\nThe assigning starts at the pos "
            << st_[i] << " of axis " << i << ", "
            << "while the dimension of this axis is "
            << Y(0)->dim(i) << ".";
        CHECK(ed_[i] > 0 && ed_[i] <= Y(0)->dim(i))
            << "\nThe assigning ends at the pos "
            << ed_[i] << " of axis " << i << ", "
            << "while the dimension of this axis is "
            << Y(0)->dim(i) << ".";
    }
}

template <class Context>
void AssignOp<Context>::RunOnDevice() {
    Setup();

    auto X_dims = Y(0)->dims();
    for (int i = 0; i < st_.size(); i++)
        X_dims[i] = ed_[i] - st_[i];
    X_.Reshape(X_dims);

    TENSOR_FROM_VECTOR(X_starts_, st_, int);
    TENSOR_FROM_VECTOR(X_dims_, X_dims, int);
    TENSOR_FROM_VECTOR(Y_strides_, Y(0)->strides(), int);

    if (XIsType(X(0), bool)) {
        RunImpl<bool>();
    } else if (XIsType(X(0), int8_t)) {
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
            "bool", "int8", "uint8", "int32", "int64",
                 "float16", "float32", "float64",
        });
    }
}

DEPLOY_CPU(Assign);
#ifdef WITH_CUDA
DEPLOY_CUDA(Assign);
#endif

OPERATOR_SCHEMA(Assign)
     /* V */
    .NumInputs(1)
     /* X */
    .NumOutputs(1);

NO_GRADIENT(Assign);

}  // namespace dragon