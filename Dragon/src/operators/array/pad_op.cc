#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/array/pad_op.h"

namespace dragon {

#define TENSOR_FROM_VEC(tensor, vec, T) \
    { \
        tensor.Reshape({ (int64_t)vec.size() }); \
        auto* data = tensor.template mutable_data<T, CPUContext>(); \
        for (int i = 0; i < vec.size(); i++) data[i] = (T)vec[i]; \
    }

template <class Context> template <typename T>
void PadOp<Context>::ConstRunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::ConstPad(
        Y(0)->count(),
        Y(0)->ndim(),
        X_dims_.template data<int, Context>(),
        X_strides_.template data<int, Context>(),
        Y_dims_.template data<int, Context>(),
        pads_.template data<int, Context>(),
        value_, x, y, ctx()
    );
}

template <class Context> template <typename T>
void PadOp<Context>::ReflectRunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::ReflectPad(
        Y(0)->count(),
        Y(0)->ndim(),
        X_dims_.template data<int, Context>(),
        X_strides_.template data<int, Context>(),
        Y_dims_.template data<int, Context>(),
        pads_.template data<int, Context>(),
        x, y, ctx()
    );
}

template <class Context> template <typename T>
void PadOp<Context>::EdgeRunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::EdgePad(
        Y(0)->count(),
        Y(0)->ndim(),
        X_dims_.template data<int, Context>(),
        X_strides_.template data<int, Context>(),
        Y_dims_.template data<int, Context>(),
        pads_.template data<int, Context>(),
        x, y, ctx()
    );
}

template <class Context> template <typename T>
void PadOp<Context>::RunImpl() {
    if (mode_ == "CONSTANT") {
        ConstRunImpl<T>();
    } else if (mode_ == "REFLECT") {
        for (int axis = 0; axis < X(0).ndim(); ++axis) {
            CHECK_LE(pad_l_[axis], X(0).dim(axis) + 1)
                << "\nThe dimension of axis " << axis
                << " is " << X(0).dim(axis) << ","
                << "\nwhile the excepted bounds of pad_l "
                << "for reflecting are (0, "
                << X(0).dim(axis) + 1 << "].";
            CHECK_LE(pad_r_[axis], X(0).dim(axis) - 1)
                << "\nThe dimension of axis " << axis
                << " is " << X(0).dim(axis) << ","
                << "\nwhile the excepted bounds of pad_r "
                << "for reflecting are (0, "
                << X(0).dim(axis) - 1 << "].";
        }
        ReflectRunImpl<T>();
    } else if (mode_ == "EDGE") {
        EdgeRunImpl<T>();
    } else {
        LOG(FATAL) << "Unknown PadMode: " << mode_ << ".";
    }
}

template <class Context>
void PadOp<Context>::RunOnDevice() {
    CHECK_EQ(X(0).ndim(), (int)pad_l_.size())
        << "\nThe padding is performed on "
        << pad_l_.size() << " dimensions, "
        << "but the num of dimensions of input is "
        << X(0).ndim() << ".";

    auto Y_dims = X(0).dims();
    for (int i = 0; i < pad_l_.size(); i++)
        Y_dims[i] += (pad_l_[i] + pad_r_[i]);
    
    Y(0)->Reshape(Y_dims);

    // Just copy the contents
    if (X(0).dims() == Y_dims) {
        Y(0)->CopyFrom(X(0), ctx());
        return;
    }

    TENSOR_FROM_VEC(pads_, pad_l_, int);
    TENSOR_FROM_VEC(X_dims_, X(0).dims(), int);
    TENSOR_FROM_VEC(X_strides_, X(0).strides(), int);
    TENSOR_FROM_VEC(Y_dims_, Y_dims, int);

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
void PadGradientOp<Context>::ConstRunImpl() {
    auto* dy = X(0).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    // Apply a simple Nd-Narrow solution
    kernel::Crop(
        Y(0)->count(),
        Y(0)->ndim(),
        Y_strides_.template data<int, Context>(),
        X_dims_.template data<int, Context>(),
        pads_.template data<int, Context>(),
        dy, dx, ctx()
    );
}

template <class Context> template <typename T>
void PadGradientOp<Context>::ReflectRunImpl() {
    LOG(FATAL) << "Not Implemented: ReflectPadGrad";
}

template <class Context> template <typename T>
void PadGradientOp<Context>::EdgeRunImpl() {
    LOG(FATAL) << "Not Implemented: EdgetPadGrad";
}

template <class Context> template <typename T>
void PadGradientOp<Context>::RunImpl() {
    if (mode_ == "CONSTANT") {
        ConstRunImpl<T>();
    } else if (mode_ == "REFLECT") {
        for (int axis = 0; axis < X(0).ndim(); ++axis) {
            CHECK_LE(pad_l_[axis], X(0).dim(axis) + 1)
                << "\nThe dimension of axis " << axis
                << " is " << X(0).dim(axis) << ","
                << "\nwhile the excepted bounds of pad_l "
                << "for reflecting are (0, "
                << X(0).dim(axis) + 1 << "].";
            CHECK_LE(pad_r_[axis], X(0).dim(axis) - 1)
                << "\nThe dimension of axis " << axis
                << " is " << X(0).dim(axis) << ","
                << "\nwhile the excepted bounds of pad_r "
                << "for reflecting are (0, "
                << X(0).dim(axis) - 1 << "].";
        }
        ReflectRunImpl<T>();
    } else if (mode_ == "EDGE") {
        EdgeRunImpl<T>();
    } else {
        LOG(FATAL) << "Unknown Mode: " << mode_ << ".";
    }
}

template <class Context>
void PadGradientOp<Context>::RunOnDevice() {
    CHECK_EQ(X(0).ndim(), (int)pad_l_.size())
        << "\nThe padding is performed on "
        << pad_l_.size() << " dimensions, "
        << "but the number of dimensions of input is "
        << X(0).ndim() << ".";

    auto X_dims = X(-1).dims();
    for (int i = 0; i < pad_l_.size(); i++)
        X_dims[i] -= (pad_l_[i] + pad_r_[i]);

    Y(0)->Reshape(X_dims);

    // Just copy the contents
    if (X(-1).dims() == X_dims) {
        Y(0)->CopyFrom(X(-1), ctx());
        return;
    }

    TENSOR_FROM_VEC(pads_, pad_l_, int);
    TENSOR_FROM_VEC(Y_strides_, X(0).strides(), int);
    TENSOR_FROM_VEC(X_dims_, X_dims, int);

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

DEPLOY_CPU(Pad);
#ifdef WITH_CUDA
DEPLOY_CUDA(Pad);
#endif

DEPLOY_CPU(PadGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(PadGradient);
#endif

OPERATOR_SCHEMA(Pad)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(PadGradient)
     /* dY */
    .NumInputs(1)
     /* dX */
    .NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ GO(0) }),
            vector<string>({ GI(0) })
        );
    }
};

}  // namespace

REGISTER_GRADIENT(Pad, GradientMaker);

#undef TENSOR_FROM_VEC

}  // namespace dragon