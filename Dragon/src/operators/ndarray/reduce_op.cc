#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/ndarray/reduce_op.h"

namespace dragon {

#define TENSOR_FROM_VECTOR(tensor, vec, T) \
    { \
        tensor.Reshape({ (int64_t)vec.size() }); \
        auto* data = tensor.template mutable_data<T, CPUContext>(); \
        for (int i = 0; i < vec.size(); i++) data[i] = (T)vec[i]; \
    }

template <class Context> template <typename T>
void ReduceOp<Context>::RunWithType() {
    dims = Input(0).dims();
    dims32.assign(dims.begin(), dims.end());
    axes32.assign(axes.begin(), axes.end());

    if (axes32.empty()) {
        // Reduce to a Scalar if missing axes
        for (int i = 0; i < Input(0).ndim(); ++i)
            axes32.push_back(i);
    }

    for (int i = 0; i < axes32.size(); i++) {
        int axis = axes32[i];
        axes32[i] = axis < 0 ? axis + Input(0).ndim() : axis;
        CHECK(axes32[i] >= 0 && axes32[i] < Input(0).ndim()) \
            << "\nExcepted the axis in [-" << Input(0).ndim()
            << ", " << Input(0).ndim() << "), got " << axis << ".";
        dims[axes32[i]] = 1;
    }

    vector<int64_t> y_dims;
    for (const auto& dim : dims) {
        if (dim != 1 || keep_dims)
            y_dims.emplace_back(dim);
    }

    Output(0)->Reshape(y_dims);

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    auto scale = operation == "SUM" ? 1.f : 1.f /
        (Input(0).count() / Output(0)->count());

    if (Input(0).count() == 1) {
        // Just copy the contents
        ctx()->template Copy<T, Context, Context>(
            Output(0)->count(), Ydata, Xdata);
    } else {
        kernel::ReduceSum(
            (int)dims32.size(), dims32.data(),
                (int)axes32.size(), axes32.data(),
                    scale, Xdata, Ydata, ctx());
    }
}

template <class Context>
void ReduceOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(Reduce);
#ifdef WITH_CUDA
DEPLOY_CUDA(Reduce);
#endif

OPERATOR_SCHEMA(Reduce)
    .NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void ReduceGradientOp<Context>::RunWithType() {
    y_dimsV= Input(0).dims();
    axes32.assign(axes.begin(), axes.end());

    if (axes32.empty()) {
        // Reduce to a Scalar if missing axes
        for (int i = 0; i < Input(0).ndim(); ++i)
            axes32.push_back(i);
    }

    for (int i = 0; i < axes32.size(); i++) {
        int axis = axes32[i];
        axes32[i] = axis < 0 ? axis + Input(0).ndim() : axis;
        CHECK(axes32[i] >= 0 && axes32[i] < Input(0).ndim()) \
            << "\nExcepted the axis in [-" << Input(0).ndim()
            << ", " << Input(0).ndim() << "), got " << axis << ".";
        y_dimsV[axes32[i]] = 1;
    }

  
    Output(0)->ReshapeLike(Input(0));

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    auto scale = operation == "SUM" ? 1.f : 1.f /
        (Input(0).count() / Input(-1).count());

    if (Input(0).count() == 1) {
        // Just copy the contents
        ctx()->template Copy<T, Context, Context>(
            Output(0)->count(), dXdata, dYdata);
    } else if (Input(-1).count() == 1) {
        // Directly set the dX from a constant Scalar
        T dYHost = Input(-1).template data<T, CPUContext>()[0];
        dYHost = cast::to<T>(cast::to<float>(dYHost) * scale);
        math::Set(Output(0)->count(), dYHost, dXdata, ctx());
    } else {
        // We need a unsqueezed strides
        int64_t stride = 1;
        y_stridesV.resize(y_dimsV.size(), 1);
        for (int i = (int)y_dimsV.size() - 1; i >= 0; i--) {
            y_stridesV[i] = stride; stride *= y_dimsV[i];
        }

        TENSOR_FROM_VECTOR(x_dimsT, Input(0).dims(), int);
        TENSOR_FROM_VECTOR(y_dimsT, y_dimsV, int);
        TENSOR_FROM_VECTOR(y_stridesT, y_stridesV, int);

        auto* XDS = x_dimsT.template data<int, Context>();
        auto* YDS = y_dimsT.template data<int, Context>();
        auto* YSS = y_stridesT.template data<int, Context>();

        // Apply a simple Nd-Broadcast solution
        kernel::ReduceSumGrad(Output(0)->count(), Output(0)->ndim(),
            XDS, YDS, YSS, scale, dYdata, dXdata, ctx());
    }
}

template <class Context>
void ReduceGradientOp<Context>::RunOnDevice() {
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

DEPLOY_CPU(ReduceGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ReduceGradient);
#endif

OPERATOR_SCHEMA(ReduceGradient)
    .NumInputs(2).NumOutputs(1);

REGISTER_GRADIENT(Reduce, SimpleGradientMaker);

}  // namespace dragon