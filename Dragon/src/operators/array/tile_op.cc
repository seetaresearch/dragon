#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/array/tile_op.h"

namespace dragon {

#define PUT_X_AND_Y(X, Y) \
    const T* X; T* Y; \
    if (src == &nav) { \
        X = ws()->template caches<T, Context> \
            ({ src->count() })[0]; \
    } else { \
        X = src->template data<T, Context>(); \
    } \
    if (dst == &nav) { \
        Y = ws()->template caches<T, Context> \
            ({ dst->count() })[0]; \
    } else { \
        Y = dst->template mutable_data<T, Context>(); \
    }

#define TENSOR_FROM_VECTOR(tensor, vec, T) \
    { \
        tensor.Reshape({ (int64_t)vec.size() }); \
        auto* data = tensor.template mutable_data<T, CPUContext>(); \
        for (int i = 0; i < vec.size(); i++) data[i] = (T)vec[i]; \
    }

template <class Context> template <typename T>
void TileOp<Context>::RunWithType() {
    auto* XDS = x_dimsT.template data<int, Context>();
    auto* YDS = y_dimsT.template data<int, Context>();
    auto* XSS = x_stridesT.template data<int, Context>();

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    // Apply a simple Nd-Broadcast solution
    kernel::Tile(Output(0)->count(), Output(0)->ndim(),
        XDS, XSS, YDS, Xdata, Ydata, ctx());
}

template <class Context>
void TileOp<Context>::RunOnDevice() {
    y_dimsV = Input(0).dims();

    for (int i = 0; i < y_dimsV.size(); ++i) {
        y_dimsV[i] *= multiples(i);
    } Output(0)->Reshape(y_dimsV);

    // Just copy the contents
    if (Input(0).dims() == y_dimsV) {
        Output(0)->template CopyFrom<Context>(
            Input(0), ctx()); return;
    }

    TENSOR_FROM_VECTOR(x_stridesT, Input(0).strides(), int);
    TENSOR_FROM_VECTOR(x_dimsT, Input(0).dims(), int);
    TENSOR_FROM_VECTOR(y_dimsT, y_dimsV, int);

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

DEPLOY_CPU(Tile);
#ifdef WITH_CUDA
DEPLOY_CUDA(Tile);
#endif
OPERATOR_SCHEMA(Tile).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void TileGradientOp<Context>::RunWithType() {
    PUT_X_AND_Y(dYdata, dXdata);

    kernel::TileGrad(rows, cols,
        multiple, dYdata, dXdata, ctx());
}

template <class Context>
void TileGradientOp<Context>::RunOnDevice() {
    // Add the axes
    vector< pair<int, int> > dispatch_axes;
    for (int i = 0; i < Input(0).ndim(); i++) {
        auto m = multiples(i);
        if (m > 1) { dispatch_axes.push_back({ m, i }); }
    }
    std::sort(dispatch_axes.begin(), dispatch_axes.end());
    std::reverse(dispatch_axes.begin(), dispatch_axes.end());

    // Just copy the contents
    if (dispatch_axes.size() == 0) {
        Output(0)->ReshapeLike(Input(-1));
        Output(0)->template CopyFrom<Context>(
            Input(-1), ctx()); return;
    }

    // Select the src && dst
    src = &Input(-1); dst = Output(0);
    if (dispatch_axes.size() % 2 == 0) dst = &nav;

    for (const auto& task : dispatch_axes) {
        axis = task.second; multiple = task.first;

        auto dims = src->dims();
        dims[axis] /= multiple;
        dst->Reshape(dims);
        rows = dst->count(0, axis);
        cols = dst->count(axis);

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
        }); ctx()->FinishDeviceCompution();

        // Protect X if num_axes >= 2
        std::swap(src, dst);
        if (dispatch_axes.size() % 2 == 1) {
            if (dst == &Input(-1)) dst = &nav;
        } else {
            if (dst == &Input(-1)) dst = Output(0);
        }
    }
}

DEPLOY_CPU(TileGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(TileGradient);
#endif

OPERATOR_SCHEMA(TileGradient)
    .NumInputs(1).NumOutputs(1);

class GetTileGradient final : public GradientMakerBase { 
 public:
    GRADIENT_MAKER_CTOR(GetTileGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ GO(0) }),
            vector<string>({ GI(0) }));
    }
};

REGISTER_GRADIENT(Tile, GetTileGradient);

#undef PUT_X_AND_Y
#undef TENSOR_FROM_VECTOR

}  // namespace dragon