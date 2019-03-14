#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/array/slice_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGUMENTS(X) \
    axis = OperatorBase::Arg<int64_t>("axis", 0); \
    axis = axis < 0 ? axis + X.ndim() : axis; \
    CHECK(axis >= 0 && axis < X.ndim()) \
       << "\nExcepted the axis in [-" << X.ndim() << ", " << X.ndim() \
       << "), got " << OperatorBase::Arg<int64_t>("axis", 0) << "."; \
    if (slice_points.empty()) { \
        CHECK_EQ(Input(0).dim(axis) % N, 0) \
            << "\nSelected dim is " << Input(0).dim(axis) \
            << ", can't be sliced into " << N << " parts."; \
    }

template <class Context> template <typename T>
void SliceOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();

    for (int i = 0; i < N; i++) {
        if (!slice_points.empty()) {
            if (i < N - 1) { y_slice_dim = slice_points[i] - slice_offset; }
            else { y_slice_dim = Input(0).dim(axis) - slice_offset; }
        }

        CHECK(y_slice_dim > 0 && slice_offset + y_slice_dim <= x_slice_dim)
            << "\nIllegal slice points: " << Tensor::DimString(slice_points)
            << " for dimension " << Input(0).dim(axis) << ".";

        slice_dims[axis] = y_slice_dim;
        Output(i)->Reshape(slice_dims);

        auto* Ydata = Output(i)->template mutable_data<T, Context>();

        kernel::Slice(
            outer_dim, inner_dim,
                x_slice_dim, y_slice_dim,
                    slice_offset, Xdata, Ydata, ctx());

        slice_offset += y_slice_dim;
    }
}

template <class Context>
void SliceOp<Context>::RunOnDevice() {
    N = OutputSize(), slice_offset = 0;
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    x_slice_dim = Input(0).dim(axis);
    y_slice_dim = x_slice_dim / N;
    slice_dims = Input(0).dims();

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

DEPLOY_CPU(Slice);
#ifdef WITH_CUDA
DEPLOY_CUDA(Slice);
#endif
OPERATOR_SCHEMA(Slice).NumInputs(1).NumOutputs(1, INT_MAX);

template <class Context> template <typename T>
void SliceGradientOp<Context>::RunWithType() {
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    for (int i = 0; i < N; i++) {
        if (!slice_points.empty()) {
            if (i < N - 1) { y_slice_dim = slice_points[i] - slice_offset; }
            else { y_slice_dim = Input(0).dim(axis) - slice_offset; }
        }

        CHECK(y_slice_dim > 0 && slice_offset + y_slice_dim <= x_slice_dim)
            << "\nIllegal slice points: " << Tensor::DimString(slice_points)
            << " for dimension " << Input(0).dim(axis) << ".";

        const T* dYdata = Input(i + 1).name() != "ignore" ?
            Input(i + 1).template data<T, Context>() : nullptr;

        kernel::SliceGrad(
            outer_dim, inner_dim,
                x_slice_dim, y_slice_dim,
                    slice_offset, dYdata, dXdata, ctx());

        slice_offset += y_slice_dim;
    }
}

template <class Context>
void SliceGradientOp<Context>::RunOnDevice() {
    N = InputSize() - 1, slice_offset = 0;
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));

    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    x_slice_dim = Input(0).dim(axis);
    y_slice_dim = x_slice_dim / N;

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

DEPLOY_CPU(SliceGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SliceGradient);
#endif

OPERATOR_SCHEMA(SliceGradient)
    .NumInputs(2, INT_MAX).NumOutputs(1);

class GetSliceGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetSliceGradient);
    vector<OperatorDef> MakeDefs() override {
        vector<string> inputs({ I(0) });
        for (int i = 0; i < def.output_size(); i++)
            inputs.push_back(GO(i));
        return SingleDef(def.type() + "Gradient", "",
            inputs, vector<string>({GI(0)}));
    }
};

REGISTER_GRADIENT(Slice, GetSliceGradient);

#undef DETERMINE_RUNTIME_ARGUMENTS

}  // namespace dragon