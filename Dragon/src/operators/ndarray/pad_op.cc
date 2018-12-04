#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/ndarray/pad_op.h"

namespace dragon {

template <class Context> template <typename T>
void PadOp<Context>::ConstRunWithType() {
    const T* Xdata; T* Ydata;
    if (source == &navigator) {
        Xdata = ws()->template caches<T, Context>({ source->count() })[0];
    } else { Xdata = source->template data<T, Context>(); }
    if (dest == &navigator) {
        Ydata = ws()->template caches<T, Context>({ dest->count() })[0];
    } else { Ydata = dest->template mutable_data<T, Context>(); }

    kernel::ConstPad1d<T, Context>(dest->count(),
        dim, dim + pad_l[axis] + pad_r[axis], inner_dim,
            pad_l[axis], value, Xdata, Ydata, ctx());
}

template <class Context> template <typename T>
void PadOp<Context>::ReflectRunWithType() {
    const T* Xdata; T* Ydata;
    if (source == &navigator) {
        Xdata = ws()->template caches<T, Context>({ source->count() })[0];
    } else { Xdata = source->template data<T, Context>(); }
    if (dest == &navigator) {
        Ydata = ws()->template caches<T, Context>({ dest->count() })[0];
    } else { Ydata = dest->template mutable_data<T, Context>(); }

    kernel::ReflectPad1d<T, Context>(dest->count(),
        dim, dim + pad_l[axis] + pad_r[axis], inner_dim,
            pad_l[axis], Xdata, Ydata, ctx());
}

template <class Context> template <typename T>
void PadOp<Context>::EdgeRunWithType() {
    const T* Xdata; T* Ydata;
    if (source == &navigator) {
        Xdata = ws()->template caches<T, Context>({ source->count() })[0];
    } else { Xdata = source->template data<T, Context>(); }
    if (dest == &navigator) {
        Ydata = ws()->template caches<T, Context>({ dest->count() })[0];
    } else { Ydata = dest->template mutable_data<T, Context>(); }

    kernel::EdgePad1d<T, Context>(dest->count(),
        dim, dim + pad_l[axis] + pad_r[axis], inner_dim,
            pad_l[axis], Xdata, Ydata, ctx());
}

template <class Context>
void PadOp<Context>::RunOnDevice() {
    CHECK_EQ(Input(0).ndim(), pad_l.size())
        << "\nThe padding is performed on "
        << pad_l.size() << " dimensions, "
        << "but the num of dimensions of input is "
        << Input(0).ndim() << ".";

    // Do nothing
    if (process_axes.size() == 0) {
        Output(0)->ReshapeLike(Input(0));
        Output(0)->template CopyFrom<Context>(Input(0), ctx());
        return;
    }

    // Select source & dest
    source = &Input(0);
    if (process_axes.size() % 2 == 1) dest = Output(0);
    else dest = &navigator;

    for (auto& task : process_axes) {
        axis = task.second;
        vector<TIndex> dims = source->dims();
        inner_dim = source->count(axis + 1);
        dim = source->dim(axis);
        dims[axis] += (pad_l[axis] + pad_r[axis]);
        dest->Reshape(dims);
        if (mode == "CONSTANT") {
            if (XIsType(Input(0), float)) ConstRunWithType<float>();
            else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
        } else if (mode == "REFLECT") {
            CHECK_LE(pad_l[axis], dim + 1)
                << "\nThe dimension of axis " << axis
                << " is " << dim << ","
                << "\nwhile the excepted bounds of pad_l "
                << "for reflecting are (0, " << dim + 1 << "].";
            CHECK_LE(pad_r[axis], dim - 1)
                << "\nThe dimension of axis " << axis
                << " is " << dim << ","
                << "\nwhile the excepted bounds of pad_r "
                << "for reflecting are (0, " << dim - 1 << "].";
            if (XIsType(Input(0), float)) ReflectRunWithType<float>();
            else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
        } else if (mode == "EDGE")  {
            if (XIsType(Input(0), float)) EdgeRunWithType<float>();
            else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
        } else {
            LOG(FATAL) << "Unsupported padding mode: " << mode << ".";
        }
        ctx()->FinishDeviceCompution();
        // Allow buffer to protect X if the num of tasks >= 2
        std::swap(source, dest);
        if (process_axes.size() % 2 == 1) {
            if (dest == &Input(0)) dest = &navigator;
        } else {
            if (dest == &Input(0)) dest = Output(0);
        }
    }
}

DEPLOY_CPU(Pad);
#ifdef WITH_CUDA
DEPLOY_CUDA(Pad);
#endif
OPERATOR_SCHEMA(Pad).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void PadGradientOp<Context>::ConstRunWithType() {
    const T* dYdata; T* dXdata;
    if (source == &navigator) {
        dYdata = ws()->template caches<T, Context>({ source->count() })[0];
    } else { dYdata = source->template data<T, Context>(); }
    if (dest == &navigator) {
        dXdata = ws()->template caches<T, Context>({ dest->count() })[0];
    } else { dXdata = dest->template mutable_data<T, Context>(); }

    kernel::ConstPad1dGrad<T, Context>(dest->count(),
        dim - pad_l[axis] - pad_r[axis], dim, inner_dim,
            pad_l[axis], dYdata, dXdata, ctx());
}

template <class Context> template <typename T>
void PadGradientOp<Context>::ReflectRunWithType() {
    const T* dYdata; T* dXdata;
    if (source == &navigator) {
        dYdata = ws()->template caches<T, Context>({ source->count() })[0];
    } else { dYdata = source->template data<T, Context>(); }
    if (dest == &navigator) {
        dXdata = ws()->template caches<T, Context>({ dest->count() })[0];
    } else { dXdata = dest->template mutable_data<T, Context>(); }

    math::Set<T, Context>(dest->count(), 0, dXdata, ctx());

    kernel::ReflectPad1dGrad<T, Context>(source->count(),
        dim - pad_l[axis] - pad_r[axis], dim, inner_dim,
            pad_l[axis], dYdata, dXdata, ctx());
}

template <class Context> template <typename T>
void PadGradientOp<Context>::EdgeRunWithType() {
    const T* dYdata; T* dXdata;
    if (source == &navigator) {
        dYdata = ws()->template caches<T, Context>({ source->count() })[0];
    } else { dYdata = source->template data<T, Context>(); }
    if (dest == &navigator) {
        dXdata = ws()->template caches<T, Context>({ dest->count() })[0];
    } else { dXdata = dest->template mutable_data<T, Context>(); }

    math::Set<T, Context>(dest->count(), 0, dXdata, ctx());

    kernel::EdgePad1dGrad<T, Context>(source->count(),
        dim - pad_l[axis] - pad_r[axis], dim, inner_dim,
            pad_l[axis], dYdata, dXdata, ctx());
}

template <class Context>
void PadGradientOp<Context>::RunOnDevice() {
    CHECK_EQ(Input(0).ndim(), pad_l.size())
        << "\nThe padding is performed on "
        << pad_l.size() << " dimensions, "
        << "but the number of dimensions of input is "
        << Input(0).ndim() << ".";

    // Do nothing
    if (process_axes.size() == 0) {
        Output(0)->ReshapeLike(Input(-1));
        Output(0)->template CopyFrom<Context>(Input(-1), ctx());
        return;
    }

    // Select source & buffer
    source = &Input(-1);
    if (process_axes.size() % 2 == 1) dest = Output(0);
    else dest = &navigator;

    for (auto& task : process_axes) {
        axis = task.second;
        vector<TIndex> dims = source->dims();
        inner_dim = source->count(axis + 1);
        dim = source->dim(axis);
        dims[axis] -= (pad_l[axis] + pad_r[axis]);
        dest->Reshape(dims);
        if (mode == "CONSTANT") {
            if (XIsType(Input(0), float)) ConstRunWithType<float>();
            else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
        } else if (mode == "REFLECT") {
            CHECK_LE(pad_l[axis], dim + 1)
                << "\nThe dimension of axis " << axis
                << " is " << dim << ","
                << "\nwhile the excepted bounds of pad_l "
                << "for reflecting are (0, " << dim + 1 << "].";
            CHECK_LE(pad_r[axis], dim - 1)
                << "\nThe dimension of axis " << axis 
                << " is " << dim << ","
                << "\nwhile the excepted bounds of pad_r "
                << "for reflecting are (0, " << dim - 1 << "].";
            if (XIsType(Input(0), float)) ReflectRunWithType<float>();
            else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
        } else if (mode == "EDGE")  {
            if (XIsType(Input(0), float)) EdgeRunWithType<float>();
            else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
        } else {
            LOG(FATAL) << "Unsupported padding mode: " << mode << ".";
        }
        ctx()->FinishDeviceCompution();
        // Allow buffer to protect X if the num of tasks >= 2
        std::swap(source, dest);
        if (process_axes.size() % 2 == 1) {
            if (dest == &Input(-1)) dest = &navigator;
        } else {
            if (dest == &Input(-1)) dest = Output(0);
        }
    }
}

DEPLOY_CPU(PadGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(PadGradient);
#endif
OPERATOR_SCHEMA(PadGradient).NumInputs(1).NumOutputs(1);

class GetPadGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetPadGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Pad, GetPadGradient);

}  // namespace dragon