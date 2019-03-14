#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/array/pad_op.h"

namespace dragon {

#define TENSOR_FROM_VECTOR(tensor, vec, T) \
    { \
        tensor.Reshape({ (int64_t)vec.size() }); \
        auto* data = tensor.template mutable_data<T, CPUContext>(); \
        for (int i = 0; i < vec.size(); i++) data[i] = (T)vec[i]; \
    }

template <class Context> template <typename T>
void PadOp<Context>::ConstRunWithType() {
    auto* XDS = x_dimsT.template data<int, Context>();
    auto* XSS = x_stridesT.template data<int, Context>();
    auto* YDS = y_dimsT.template data<int, Context>();
    auto* LPS = l_padsT.template data<int, Context>();

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::ConstPad(Output(0)->count(), Output(0)->ndim(),
        XDS, XSS, YDS, LPS, value, Xdata, Ydata, ctx());
}

template <class Context> template <typename T>
void PadOp<Context>::ReflectRunWithType() {
    auto* XDS = x_dimsT.template data<int, Context>();
    auto* XSS = x_stridesT.template data<int, Context>();
    auto* YDS = y_dimsT.template data<int, Context>();
    auto* LPS = l_padsT.template data<int, Context>();

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::ReflectPad(Output(0)->count(), Output(0)->ndim(),
        XDS, XSS, YDS, LPS, Xdata, Ydata, ctx());
}

template <class Context> template <typename T>
void PadOp<Context>::EdgeRunWithType() {
    auto* XDS = x_dimsT.template data<int, Context>();
    auto* XSS = x_stridesT.template data<int, Context>();
    auto* YDS = y_dimsT.template data<int, Context>();
    auto* LPS = l_padsT.template data<int, Context>();

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::EdgePad(Output(0)->count(), Output(0)->ndim(),
        XDS, XSS, YDS, LPS, Xdata, Ydata, ctx());
}

template <class Context> template <typename T>
void PadOp<Context>::RunWithType() {
    if (mode == "CONSTANT") {
        ConstRunWithType<T>();
    } else if (mode == "REFLECT") {
        for (int axis = 0; axis < Input(0).ndim(); ++axis) {
            CHECK_LE(pad_l[axis], Input(0).dim(axis) + 1)
                << "\nThe dimension of axis " << axis
                << " is " << Input(0).dim(axis) << ","
                << "\nwhile the excepted bounds of pad_l "
                << "for reflecting are (0, " << Input(0).dim(axis) + 1 << "].";
            CHECK_LE(pad_r[axis], Input(0).dim(axis) - 1)
                << "\nThe dimension of axis " << axis
                << " is " << Input(0).dim(axis) << ","
                << "\nwhile the excepted bounds of pad_r "
                << "for reflecting are (0, " << Input(0).dim(axis) - 1 << "].";
        }
        ReflectRunWithType<T>();
    } else if (mode == "EDGE") {
        EdgeRunWithType<T>();
    } else {
        LOG(FATAL) << "Unsupported padding mode: " << mode << ".";
    }
}

template <class Context>
void PadOp<Context>::RunOnDevice() {
    CHECK_EQ(Input(0).ndim(), (int)pad_l.size())
        << "\nThe padding is performed on "
        << pad_l.size() << " dimensions, "
        << "but the num of dimensions of input is "
        << Input(0).ndim() << ".";

    y_dimsV = Input(0).dims();
    for (int i = 0; i < pad_l.size(); i++) {
        y_dimsV[i] += pad_l[i] + pad_r[i];
    } Output(0)->Reshape(y_dimsV);

    // Just copy the contents
    if (Input(0).dims() == y_dimsV) {
        Output(0)->template CopyFrom<Context>(
            Input(0), ctx()); return;
    }
    
    TENSOR_FROM_VECTOR(x_dimsT, Input(0).dims(), int);
    TENSOR_FROM_VECTOR(x_stridesT, Input(0).strides(), int);
    TENSOR_FROM_VECTOR(y_dimsT, y_dimsV, int);
    TENSOR_FROM_VECTOR(l_padsT, pad_l, int);

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

DEPLOY_CPU(Pad);
#ifdef WITH_CUDA
DEPLOY_CUDA(Pad);
#endif
OPERATOR_SCHEMA(Pad).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void PadGradientOp<Context>::ConstRunWithType() {
    auto* YSS = y_stridesT.template data<int, Context>();
    auto* XDS = x_dimsT.template data<int, Context>();
    auto* LPS = l_padsT.template data<int, Context>();

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    // Apply a simple Nd-Narrow solution
    kernel::Crop(Output(0)->count(), Output(0)->ndim(),
        YSS, XDS, LPS, dYdata, dXdata, ctx());
}

template <class Context> template <typename T>
void PadGradientOp<Context>::ReflectRunWithType() {
    LOG(FATAL) << "Not Implemented: ReflectPadGrad";
}

template <class Context> template <typename T>
void PadGradientOp<Context>::EdgeRunWithType() {
    LOG(FATAL) << "Not Implemented: EdgetPadGrad";
}

template <class Context> template <typename T>
void PadGradientOp<Context>::RunWithType() {
    if (mode == "CONSTANT") {
        ConstRunWithType<T>();
    } else if (mode == "REFLECT") {
        for (int axis = 0; axis < Input(0).ndim(); ++axis) {
            CHECK_LE(pad_l[axis], Input(0).dim(axis) + 1)
                << "\nThe dimension of axis " << axis
                << " is " << Input(0).dim(axis) << ","
                << "\nwhile the excepted bounds of pad_l "
                << "for reflecting are (0, " << Input(0).dim(axis) + 1 << "].";
            CHECK_LE(pad_r[axis], Input(0).dim(axis) - 1)
                << "\nThe dimension of axis " << axis
                << " is " << Input(0).dim(axis) << ","
                << "\nwhile the excepted bounds of pad_r "
                << "for reflecting are (0, " << Input(0).dim(axis) - 1 << "].";
        }
        ReflectRunWithType<T>();
    } else if (mode == "EDGE") {
        EdgeRunWithType<T>();
    } else {
        LOG(FATAL) << "Unsupported padding mode: " << mode << ".";
    }
}

template <class Context>
void PadGradientOp<Context>::RunOnDevice() {
    CHECK_EQ(Input(0).ndim(), (int)pad_l.size())
        << "\nThe padding is performed on "
        << pad_l.size() << " dimensions, "
        << "but the number of dimensions of input is "
        << Input(0).ndim() << ".";

    x_dimsV = Input(-1).dims();
    for (int i = 0; i < pad_l.size(); i++) {
        x_dimsV[i] -= pad_l[i] + pad_r[i];
    } Output(0)->Reshape(x_dimsV);

    // Just copy the contents
    if (Input(-1).dims() == x_dimsV) {
        Output(0)->template CopyFrom<Context>(
            Input(-1), ctx()); return;
    }

    TENSOR_FROM_VECTOR(y_stridesT, Input(-1).strides(), int);
    TENSOR_FROM_VECTOR(x_dimsT, x_dimsV, int);
    TENSOR_FROM_VECTOR(l_padsT, pad_l, int);

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

DEPLOY_CPU(PadGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(PadGradient);
#endif

OPERATOR_SCHEMA(PadGradient)
    .NumInputs(1).NumOutputs(1);

class GetPadGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetPadGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ GO(0) }),
            vector<string>({ GI(0) }));
    }
};

REGISTER_GRADIENT(Pad, GetPadGradient);

#undef TENSOR_FROM_VECTOR

}  // namespace dragon