#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/array/crop_op.h"

namespace dragon {

#define TENSOR_FROM_VECTOR(tensor, vec, T) \
    { \
        tensor.Reshape({ (int64_t)vec.size() }); \
        auto* data = tensor.template mutable_data<T, CPUContext>(); \
        for (int i = 0; i < vec.size(); i++) data[i] = (T)vec[i]; \
    }

#define WSTENSOR_FROM_VECTOR(name, vec, T) \
    { \
        auto* t = ws()->CreateTensor(mount_name(name)) \
            ->Reshape({ (int64_t)vec.size() }); \
        auto* data = t->template mutable_data<T, CPUContext>(); \
        for (int i = 0; i < vec.size(); i++) data[i] = vec[i]; \
    }

#define VECTOR_FROM_WSTENSOR(name, vec, T) \
    { \
        auto* t = ws()->GetTensor(mount_name(name)); \
        vec.assign((size_t)t->count(), 0); \
        auto* data = t->template data<T, CPUContext>(); \
        for (int i = 0; i < t->count(); i++) vec[i] = data[i]; \
    }

template <class Context> template <typename T>
void CropOp<Context>::RunWithType() {
    auto* XSS = x_stridesT.template data<int, Context>();
    auto* YDS = y_dimsT.template data<int, Context>();
    auto* STS = startsT.template data<int, Context>();

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    // Apply a simple Nd-Narrow solution
    kernel::Crop(Output(0)->count(), y_dimsT.count(),
        XSS, YDS, STS, Xdata, Ydata, ctx());
}

template <class Context>
void CropOp<Context>::Setup() {
    st.assign((size_t)Input(0).ndim(), 0);
    ed.assign(st.size(), 0);
    keep_dims.assign(st.size(), 1);

    // Determine the starts
    if (start_axis < 0) {
        // Static solution
        int n_starts = GET_ARGUMENTS_SIZE(starts);
        for (int i = 0; i < st.size(); i++) {
            if (i < n_starts) st[i] = starts(i);
        }
    } else {
        // Dynamic solution
        for (int i = 0; i < st.size(); i++) {
            if (i < start_axis || offsets.size() == 0) {
                st[i] = 0;
            } else if (i - start_axis < (int)offsets.size()) {
                st[i] = offsets[i - start_axis];
            } else {
                st[i] = offsets[offsets.size() - 1];
            }
        }
    }

    // Determine the ends
    if (start_axis < 0) {
        // Static solution
        int n_sizes = GET_ARGUMENTS_SIZE(sizes);
        for (int i = 0; i < ed.size(); i++) {
            ed[i] = Input(0).dim(i);
            if (i < n_sizes) {
                auto len = sizes(i);
                if (len > 0) { ed[i] = st[i] + len; }
                else if (len == 0) { ed[i] = st[i] + 1; keep_dims[i] = 0; }
            }
        }
    } else {
        // Dynamic solution
        for (int i = 0; i < ed.size(); i++) {
            ed[i] = Input(0).dim(i);
            if (i >= start_axis) {
                // From a known shape
                Tensor* like = ws()->GetTensor(shape_like);
                CHECK_EQ(like->ndim(), Input(0).ndim())
                    << "\nThe cropping is performed on " << like->ndim() << " dimensions, "
                    << "while the num of dimensions of input is " << Input(0).ndim() << ".";
                ed[i] = st[i] + like->dim(i);
            }
        }
    }

    // Check starts and ends
    for (int i = 0; i < st.size(); i++) {
        CHECK(st[i] >= 0 && st[i] < Input(0).dim(i))
            << "\nThe cropping starts at the pos " << st[i] << " of axis " << i << ", "
            << "while the dimension of this axis is " << Input(0).dim(i) << ".";
        CHECK(ed[i] > 0 && ed[i] <= Input(0).dim(i))
            << "\nThe cropping ends at the pos " << ed[i] << " of axis " << i << ", "
            << "while the dimension of this axis is " << Input(0).dim(i) << ".";
    }

    // Store for the gradient op
    WSTENSOR_FROM_VECTOR("crop/starts", st, int64_t);
    WSTENSOR_FROM_VECTOR("crop/ends", ed, int64_t);

    y_dimsV = Input(0).dims();
    for (int i = 0; i < st.size(); i++) {
        y_dimsV[i] = ed[i] - st[i];
    }

    // Squeeze the dimensions
    vector<int64_t> squeeze_shape;
    for (int i = 0; i < keep_dims.size(); i++)
        if (keep_dims[i]) squeeze_shape.push_back(y_dimsV[i]);
    Output(0)->Reshape(squeeze_shape);
}

template <class Context>
void CropOp<Context>::RunOnDevice() {
    Setup();

    // Just copy the contents
    if (Input(0).dims() == y_dimsV) {
        Output(0)->template CopyFrom<Context>(
            Input(0), ctx()); return;
    }

    TENSOR_FROM_VECTOR(x_stridesT, Input(0).strides(), int);
    TENSOR_FROM_VECTOR(y_dimsT, y_dimsV, int);
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

DEPLOY_CPU(Crop);
#ifdef WITH_CUDA
DEPLOY_CUDA(Crop);
#endif
OPERATOR_SCHEMA(Crop).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void CropGradientOp<Context>::RunWithType() {
    auto* XSS = x_stridesT.template data<int, Context>();
    auto* YDS = y_dimsT.template data<int, Context>();
    auto* STS = startsT.template data<int, Context>();

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    // Zero the redundant gradients
    math::Set(Output(0)->count(), cast::to<T>(0.f), dXdata, ctx());

    // Copy the dY to the right positions
    kernel::CropGrad(Input(-1).count(), y_dimsT.count(),
        XSS, YDS, STS, dYdata, dXdata, ctx());
}

template <class Context>
void CropGradientOp<Context>::RunOnDevice() {
    VECTOR_FROM_WSTENSOR("crop/starts", st, int64_t);
    VECTOR_FROM_WSTENSOR("crop/ends", ed, int64_t);

    y_dimsV = Input(0).dims();
    for (int i = 0; i < st.size(); i++) {
        y_dimsV[i] = ed[i] - st[i];
    } Output(0)->ReshapeLike(Input(0));

    // Just copy the contents
    if (Input(0).dims() == y_dimsV) {
        Output(0)->template CopyFrom<Context>(
            Input(-1), ctx()); return;
    }

    TENSOR_FROM_VECTOR(x_stridesT, Input(0).strides(), int);
    TENSOR_FROM_VECTOR(y_dimsT, y_dimsV, int);
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

DEPLOY_CPU(CropGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(CropGradient);
#endif

OPERATOR_SCHEMA(CropGradient)
    .NumInputs(2).NumOutputs(1);

REGISTER_GRADIENT(Crop, SimpleGradientMaker);

#undef TENSOR_FROM_VECTOR
#undef WSTENSOR_FROM_VECTOR
#undef VECTOR_FROM_WSTENSOR

}  // namespace dragon