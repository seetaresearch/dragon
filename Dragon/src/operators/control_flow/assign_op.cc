#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_utils.h"
#include "utils/math_functions.h"
#include "operators/control_flow/assign_op.h"

namespace dragon {

#define TENSOR_FROM_VECTOR(tensor, vec, T) \
    { \
        tensor.Reshape({ (int64_t)vec.size() }); \
        auto* data = tensor.template mutable_data<T, CPUContext>(); \
        for (int i = 0; i < vec.size(); i++) data[i] = (T)vec[i]; \
    }

template <class Context> template <typename T>
void AssignOp<Context>::RunWithType() {
    const T* Xdata = nullptr;
    auto* XDS = x_dimsT.template data<int, Context>();
    auto* YSS = y_stridesT.template data<int, Context>();
    auto* STS = startsT.template data<int, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    if (Input(0).count() < fake_x.count()) {
        int rows, cols;
        auto* WSdata = ws()->template caches
            <T, Context>({ fake_x.count() })[0];
        auto* RXdata = Input(0).template data<T, Context>();
        if (utils::IsRowwiseBroadcast(
                fake_x.dims(), Input(0).dims(),
                    &rows, &cols)) {
            math::BroadcastSet(rows, cols, 0,
                RXdata, WSdata, ctx());
        } else if (utils::IsColwiseBroadcast(
                fake_x.dims(), Input(0).dims(),
                    &rows, &cols)) {
            math::BroadcastSet(rows, cols, 1,
                RXdata, WSdata, ctx());
        } else {
            LOG(FATAL) << "Could not broadcast "
                << Input(0).DimString() << " to "
                << fake_x.DimString();
        }
        Xdata = WSdata;
    } else if (Input(0).count() == fake_x.count()) {
        Xdata = Input(0).template data<T, Context>();
    } else {
        LOG(FATAL) << "Could not assign "
            << Input(0).DimString() << " to "
            << Output(0)->DimString();
    }

    // Apply a simple Nd-Broadcast solution
    kernel::Assign(fake_x.count(), x_dimsT.count(),
        XDS, YSS, STS, Xdata, Ydata, ctx());
}

template <class Context>
void AssignOp<Context>::Setup() {
    st.assign((size_t)Output(0)->ndim(), 0);
    ed.assign(st.size(), 0);

    // Determine the starts
    int n_starts = GET_ARGUMENTS_SIZE(starts);
    for (int i = 0; i < st.size(); i++)
        if (i < n_starts) st[i] = starts(i);
 
    // Determine the ends
    int n_sizes = GET_ARGUMENTS_SIZE(sizes);
    for (int i = 0; i < ed.size(); i++) {
        ed[i] = Output(0)->dim(i);
        if (i < n_sizes) {
            auto len = sizes(i);
            if (len > 0) { ed[i] = st[i] + len; }
            else if (len == 0) { ed[i] = st[i] + 1; }
        }
    }

    // Check starts and ends
    for (int i = 0; i < st.size(); i++) {
        CHECK(st[i] >= 0 && st[i] < Output(0)->dim(i))
            << "\nThe assigning starts at the pos " << st[i] << " of axis " << i << ", "
            << "while the dimension of this axis is " << Output(0)->dim(i) << ".";
        CHECK(ed[i] > 0 && ed[i] <= Output(0)->dim(i))
            << "\nThe assigning ends at the pos " << ed[i] << " of axis " << i << ", "
            << "while the dimension of this axis is " << Output(0)->dim(i) << ".";
    }

    x_dimsV = Output(0)->dims();
    for (int i = 0; i < st.size(); i++)
        x_dimsV[i] = ed[i] - st[i];
    fake_x.Reshape(x_dimsV);
}

template <class Context>
void AssignOp<Context>::RunOnDevice() {
    Setup();

    TENSOR_FROM_VECTOR(y_stridesT, Output(0)->strides(), int);
    TENSOR_FROM_VECTOR(x_dimsT, x_dimsV, int);
    TENSOR_FROM_VECTOR(startsT, st, int);

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

DEPLOY_CPU(Assign);
#ifdef WITH_CUDA
DEPLOY_CUDA(Assign);
#endif
OPERATOR_SCHEMA(Assign).NumInputs(1).NumOutputs(1);

NO_GRADIENT(Assign);

}  // namespace dragon