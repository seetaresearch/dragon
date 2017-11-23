#include "operators/ndarray/pad_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void PadOp<Context>::ConstRunWithType() {
    auto* Xdata = source->template data<T, Context>();
    auto* Ydata = dest->template mutable_data<T, Context>();
    kernel::ConstPad1D<T, Context>(dest->count(),
                                             dim,
                 dim + pad_l[axis] + pad_r[axis],
                                       inner_dim,
                                     pad_l[axis],
                                           value,
                                           Xdata,
                                           Ydata,
                                         &ctx());
}

template <class Context> template <typename T>
void PadOp<Context>::ReflectRunWithType() {
    auto* Xdata = source->template data<T, Context>();
    auto* Ydata = dest->template mutable_data<T, Context>();
    kernel::ReflectPad1D<T, Context>(dest->count(),
                                               dim,
                   dim + pad_l[axis] + pad_r[axis],
                                         inner_dim,
                                       pad_l[axis],
                                             Xdata,
                                             Ydata,
                                           &ctx());
}

template <class Context> template <typename T>
void PadOp<Context>::EdgeRunWithType() {
    auto* Xdata = source->template data<T, Context>();
    auto* Ydata = dest->template mutable_data<T, Context>();
    kernel::EdgePad1D<T, Context>(dest->count(),
                                            dim,
                dim + pad_l[axis] + pad_r[axis],
                                      inner_dim,
                                    pad_l[axis],
                                          Xdata,
                                          Ydata,
                                        &ctx());
}

template <class Context>
void PadOp<Context>::RunOnDevice() {
    CHECK_EQ(input(0).ndim(), pad_l.size())
        << "\nThe padding is performed on " << pad_l.size() << " dimensions, "
        << "but the num of dimensions of input is " << input(0).ndim() << ".";

    //  do nothing
    if (process_axes.size() == 0) {
        output(0)->ReshapeLike(input(0));
        output(0)->Share(input(0));
        return;
    }

    //  select source & dest
    source = &input(0);
    if (process_axes.size() % 2 == 1) dest = output(0);
    else dest = ws()->GetBuffer();

    for (auto& task : process_axes) {
        axis = task.second;
        vector<TIndex> dims = source->dims();
        inner_dim = source->count(axis + 1);
        dim = source->dim(axis);
        dims[axis] += (pad_l[axis] + pad_r[axis]);
        dest->Reshape(dims);
        if (mode == "CONSTANT") {
            if (input(0).template IsType<float>()) ConstRunWithType<float>();
            else LOG(FATAL) << "Unsupported input types.";
        } else if (mode == "REFLECT") {
            CHECK_LE(pad_l[axis], dim + 1)
                << "\nThe dimension of axis " << axis << " is " << dim << ","
                << "\nwhile the excepted bounds of pad_l for reflecting are (0, " << dim + 1 << "].";
            CHECK_LE(pad_r[axis], dim - 1)
                << "\nThe dimension of axis " << axis << " is " << dim << ","
                << "\nwhile the excepted bounds of pad_r for reflecting are (0, " << dim - 1 << "].";
            if (input(0).template IsType<float>()) ReflectRunWithType<float>();
            else LOG(FATAL) << "Unsupported input types.";
        } else if (mode == "EDGE")  {
            if (input(0).template IsType<float>()) EdgeRunWithType<float>();
            else LOG(FATAL) << "Unsupported input types.";
        } else {
            LOG(FATAL) << "Unsupported padding mode: " << mode << " .";
        }
        //  allow buffer to protect X if the num of tasks >= 2
        std::swap(source, dest);
        if (process_axes.size() % 2 == 1) {
            if (dest == &input(0)) dest = ws()->GetBuffer();
        } else {
            if (dest == &input(0)) dest = output(0);
        }
    }
    ws()->ReleaseBuffer(dest);
}

DEPLOY_CPU(Pad);
#ifdef WITH_CUDA
DEPLOY_CUDA(Pad);
#endif
OPERATOR_SCHEMA(Pad).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void PadGradientOp<Context>::ConstRunWithType() {
    auto* dYdata = source->template data<T, Context>();
    auto* dXdata = dest->template mutable_data<T, Context>();
    kernel::ConstPad1DGrad<T, Context>(dest->count(),
                     dim - pad_l[axis] - pad_r[axis],
                                                 dim,
                                           inner_dim,
                                         pad_l[axis],
                                              dYdata,
                                              dXdata,
                                             &ctx());
}

template <class Context> template <typename T>
void PadGradientOp<Context>::ReflectRunWithType() {
    auto* dYdata = source->template data<T, Context>();
    auto* dXdata = dest->template mutable_data<T, Context>();
    math::Set<T, Context>(dest->count(), 0, dXdata);
    kernel::ReflectPad1DGrad<T, Context>(source->count(),
                         dim - pad_l[axis] - pad_r[axis],
                                                     dim,
                                               inner_dim,
                                             pad_l[axis],
                                                  dYdata,
                                                 dXdata);
}

template <class Context> template <typename T>
void PadGradientOp<Context>::EdgeRunWithType() {
    auto* dYdata = source->template data<T, Context>();
    auto* dXdata = dest->template mutable_data<T, Context>();
    math::Set<T, Context>(dest->count(), 0, dXdata);
    kernel::EdgePad1DGrad<T, Context>(source->count(),
                      dim - pad_l[axis] - pad_r[axis],
                                                  dim,
                                            inner_dim,
                                          pad_l[axis],
                                               dYdata,
                                               dXdata,
                                              &ctx());
}

template <class Context>
void PadGradientOp<Context>::RunOnDevice() {
    CHECK_EQ(input(0).ndim(), pad_l.size())
        << "\nThe padding is performed on " << pad_l.size() << " dimensions, "
        << "but the number of dimensions of input is " << input(0).ndim() << ".";

    //  do nothing 
    if (process_axes.size() == 0) {
        output(0)->ReshapeLike(input(-1));
        output(0)->Share(input(-1));
        return;
    }

    //  select source & buffer
    source = &input(-1);
    if (process_axes.size() % 2 == 1) dest = output(0);
    else dest = ws()->GetBuffer();

    for (auto& task : process_axes) {
        axis = task.second;
        vector<TIndex> dims = source->dims();
        inner_dim = source->count(axis + 1);
        dim = source->dim(axis);
        dims[axis] -= (pad_l[axis] + pad_r[axis]);
        dest->Reshape(dims);
        if (mode == "CONSTANT") {
            if (input(0).template IsType<float>()) ConstRunWithType<float>();
            else LOG(FATAL) << "Unsupported input types.";
        } else if (mode == "REFLECT") {
            CHECK_LE(pad_l[axis], dim + 1)
                << "\nThe dimension of axis " << axis << " is " << dim << ","
                << "\nwhile the excepted bounds of pad_l for reflecting are (0, " << dim + 1 << "].";
            CHECK_LE(pad_r[axis], dim - 1)
                << "\nThe dimension of axis " << axis << " is " << dim << ","
                << "\nwhile the excepted bounds of pad_r for reflecting are (0, " << dim - 1 << "].";
            if (input(0).template IsType<float>()) ReflectRunWithType<float>();
            else LOG(FATAL) << "Unsupported input types.";
        } else if (mode == "EDGE")  {
            if (input(0).template IsType<float>()) EdgeRunWithType<float>();
            else LOG(FATAL) << "Unsupported input types.";
        } else {
            LOG(FATAL) << "Unsupported padding mode: " << mode << " .";
        }
        //  allow buffer to protect X if the num of tasks >= 2
        std::swap(source, dest);
        if (process_axes.size() % 2 == 1) {
            if (dest == &input(-1)) dest = ws()->GetBuffer();
        } else {
            if (dest == &input(-1)) dest = output(0);
        }
    }
    ws()->ReleaseBuffer(dest);
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

}    // namespace dragon