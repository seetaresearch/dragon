#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/moments_op.h"

namespace dragon {

template <class Context> template <typename Tx, typename Ty>
void MomentsOp<Context>::RunWithType() {
    dims = Input(0).dims(); axes32.clear();
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

    Output(0)->Reshape(y_dims); Output(1)->Reshape(y_dims);

    auto* Xdata = Input(0).template data<Tx, Context>();
    auto* Mdata = Output(0)->template mutable_data<Ty, Context>();
    auto* Vdata = Output(1)->template mutable_data<Ty, Context>();

    if (Input(0).count() == 1) {
        kernel::TypeA2B(Output(0)->count(), Xdata, Mdata, ctx());
        math::Set(Output(0)->count(), cast::to<Ty>(0.f), Vdata, ctx());
    } else {
        kernel::Moments(
            (int)dims32.size(), dims32.data(),
                (int)axes32.size(), axes32.data(),
                    Xdata, Mdata, Vdata, ctx());
    }
}

template <class Context>
void MomentsOp<Context>::RunOnDevice() {
    if (XIsType(Input(0), int8_t)) RunWithType<int8_t, float>();
    else if (XIsType(Input(0), uint8_t)) RunWithType<uint8_t, float>();
    else if (XIsType(Input(0), int)) RunWithType<int, float>();
    else if (XIsType(Input(0), int64_t)) RunWithType<int64_t, float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16, float>();
    else if (XIsType(Input(0), float)) RunWithType<float, float>();
    else if (XIsType(Input(0), double)) RunWithType<double, double>();
    else LOG(FATAL) << DTypeHelper(Input(0), {
        "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
    });
}

DEPLOY_CPU(Moments);
#ifdef WITH_CUDA
DEPLOY_CUDA(Moments);
#endif
OPERATOR_SCHEMA(Moments).NumInputs(1).NumOutputs(2);

NO_GRADIENT(Moments);

}  // namespace dragon