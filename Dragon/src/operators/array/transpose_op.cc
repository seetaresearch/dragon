#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/array/transpose_op.h"

namespace dragon {

template <class Context> template <typename T>
void TransposeOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::Transpose(
        Y(0)->count(),
        Y(0)->ndim(),
        X_strides_.template data<int, Context>(),
        Y_dims_.template data<int, Context>(),
        x, y, ctx()
    );
}

template <class Context>
void TransposeOp<Context>::RunOnDevice() {
    int num_axes = GET_ARGS_SIZE(perm);

    if (num_axes == 0) {
        // Reverse dimensions directly if missing perms
        perm_.clear(); num_axes = X(0).ndim();
        for (int i = num_axes - 1; i >= 0; i--)
            perm_.push_back(i);
    }

    CHECK_EQ(X(0).ndim(), num_axes)
        << "\nProviding " << num_axes
        << " dimensions to permute, "
        << "while Tensor(" << X(0).name()
        << ")'s dims are " << X(0).DimString();

    auto* x_strides = X_strides_
        .Reshape({ num_axes })
        ->template mutable_data<int, CPUContext>();

    auto* y_dims = Y_dims_
        .Reshape({ num_axes })
        ->template mutable_data<int, CPUContext>();

    vec64_t out_shape;

    for (int i = 0; i < num_axes; i++) {
        auto axis = perm(i);
        out_shape.push_back(X(0).dim(axis));
        x_strides[i] = (int)X(0).stride(axis);
        y_dims[i] = (int)out_shape.back();
    }

    Y(0)->Reshape(out_shape);

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

template <class Context> template <typename T>
void TransposeGradientOp<Context>::RunImpl() {
    auto* dy = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    kernel::TransposeGrad(
        Y(0)->count(),
        Y(0)->ndim(),
        X_strides_.template data<int, Context>(),
        Y_dims_.template data<int, Context>(),
        dy, dx, ctx()
    );
}

template <class Context>
void TransposeGradientOp<Context>::RunOnDevice() {
    int num_axes = GET_ARGS_SIZE(perm);

    if (num_axes == 0) {
        // Reverse dimensions directly if missing perms
        perm_.clear(); num_axes = X(0).ndim();
        for (int i = num_axes - 1; i >= 0; i--)
            perm_.push_back(i);
    }

    CHECK_EQ(X(0).ndim(), num_axes)
        << "\nProviding " << num_axes
        << " dimensions to permute, "
        << "while Tensor(" << X(0).name()
        << ")'s dims are " << X(0).DimString();

    auto* x_strides = X_strides_
        .Reshape({ num_axes })
        ->template mutable_data<int, CPUContext>();

    auto* y_dims = Y_dims_
        .Reshape({ num_axes })
        ->template mutable_data<int, CPUContext>();

    for (int i = 0; i < num_axes; i++) {
        auto axis = perm(i);
        x_strides[i] = (int)X(0).stride(axis);
        y_dims[i] = (int)X(1).dim(i);
    }

    Y(0)->ReshapeLike(X(0));

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

DEPLOY_CPU(Transpose);
#ifdef WITH_CUDA
DEPLOY_CUDA(Transpose);
#endif

DEPLOY_CPU(TransposeGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(TransposeGradient);
#endif

OPERATOR_SCHEMA(Transpose)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(TransposeGradient)
     /* X, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Transpose, SimpleGradientMaker);

}  // namespace dragon