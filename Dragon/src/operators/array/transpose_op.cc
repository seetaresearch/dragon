#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/array/transpose_op.h"

namespace dragon {

template <class Context> template <typename T>
void TransposeOp<Context>::RunWithType() {
    auto* XSS = x_strides.template data<int, Context>();
    auto* YDS = y_dims.template data<int, Context>();
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::Transpose(
        Output(0)->count(), Output(0)->ndim(),
            XSS, YDS, Xdata, Ydata, ctx());
}

template <class Context>
void TransposeOp<Context>::RunOnDevice() {
    auto given_num_axes = (int)std::max(
        perm_desc.size(), perm_value.size());
    if (given_num_axes == 0) {
        // Reverse dimensions directly if missing perms
        perm_value.clear(); given_num_axes = Input(0).ndim();
        for (int i = given_num_axes - 1; i >= 0; i--)
            perm_value.push_back(i);
    }

    CHECK_EQ(Input(0).ndim(), given_num_axes)
        << "\nProvide " << given_num_axes << " dimensions to permute, "
        << "but Tensor(" << Input(0).name() << ")'s dims are "
        << Input(0).DimString();

    x_strides.Reshape({ given_num_axes });
    y_dims.Reshape({ given_num_axes });

    auto* XSS = x_strides.template mutable_data<int, CPUContext>();
    auto* YDS = y_dims.template mutable_data<int, CPUContext>();

    vector<int64_t> output_dims;

    for (int i = 0; i < given_num_axes; i++) {
        auto axis = perm(i);
        output_dims.push_back(Input(0).dim(axis));
        XSS[i] = (int)Input(0).stride(axis);
        YDS[i] = (int)output_dims.back();
    }

    Output(0)->Reshape(output_dims);

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

DEPLOY_CPU(Transpose);
#ifdef WITH_CUDA
DEPLOY_CUDA(Transpose);
#endif
OPERATOR_SCHEMA(Transpose).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void TransposeGradientOp<Context>::RunWithType() {
    auto* XSS = x_strides.template data<int, Context>();
    auto* YDS = y_dims.template data<int, Context>();
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    kernel::TransposeGrad(
        Output(0)->count(), Output(0)->ndim(),
            XSS, YDS, dYdata, dXdata, ctx());
}

template <class Context>
void TransposeGradientOp<Context>::RunOnDevice() {
    auto given_num_axes = (int)std::max(
        perm_desc.size(), perm_value.size());

    if (given_num_axes == 0) {
        // Reverse dimensions directly if missing perms
        perm_value.clear(); given_num_axes = Input(0).ndim();
        for (int i = given_num_axes - 1; i >= 0; i--)
            perm_value.push_back(i);
    }

    CHECK_EQ(Input(0).ndim(), given_num_axes)
        << "\nProvide " << given_num_axes << " dimensions to permute, "
        << "but Tensor(" << Input(0).name() << ")'s dims are "
        << Input(0).DimString();

    x_strides.Reshape({ given_num_axes });
    y_dims.Reshape({ given_num_axes });

    auto* XSS = x_strides.template mutable_data<int, CPUContext>();
    auto* YDS = y_dims.template mutable_data<int, CPUContext>();

    for (int i = 0; i < given_num_axes; i++) {
        auto axis = perm(i);
        XSS[i] = (int)Input(0).stride(axis);
        YDS[i] = (int)Input(-1).dim(i);
    }

    Output(0)->ReshapeLike(Input(0));

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

DEPLOY_CPU(TransposeGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(TransposeGradient);
#endif

OPERATOR_SCHEMA(TransposeGradient)
    .NumInputs(2).NumOutputs(1);

REGISTER_GRADIENT(Transpose, SimpleGradientMaker);

}  // namespace dragon