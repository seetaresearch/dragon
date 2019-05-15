#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/array/crop_op.h"

namespace dragon {

#define TENSOR_FROM_VEC(tensor, vec, T) \
    { \
        tensor.Reshape({ (int64_t)vec.size() }); \
        auto* data = tensor.template mutable_data<T, CPUContext>(); \
        for (int i = 0; i < vec.size(); i++) data[i] = (T)vec[i]; \
    }

#define WSTENSOR_FROM_VEC(name, vec, T) \
    { \
        auto* t = ws()->CreateTensor(unique_name(name)) \
            ->Reshape({ (int64_t)vec.size() }); \
        auto* data = t->template mutable_data<T, CPUContext>(); \
        for (int i = 0; i < vec.size(); i++) data[i] = vec[i]; \
    }

#define VEC_FROM_WSTENSOR(name, vec, T) \
    { \
        auto* t = ws()->GetTensor(unique_name(name)); \
        vec.assign((size_t)t->count(), 0); \
        auto* data = t->template data<T, CPUContext>(); \
        for (int i = 0; i < t->count(); i++) vec[i] = data[i]; \
    }

template <class Context> template <typename T>
void CropOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    // Apply a simple Nd-Narrow solution
    kernel::Crop(
        Y(0)->count(),
        X(0).ndim(),
        X_strides_.template data<int, Context>(),
        Y_dims_.template data<int, Context>(),
        X_starts_.template data<int, Context>(),
        x, y, ctx()
    );
}

template <class Context>
void CropOp<Context>::Setup() {
    st_.assign((size_t)X(0).ndim(), 0);
    ed_.assign(st_.size(), 0);
    keep_.assign(st_.size(), 1);

    // Determine the starts
    if (start_axis_ < 0) {
        // Static solution
        int nstarts = GET_ARGS_SIZE(starts);
        for (int i = 0; i < st_.size(); i++) {
            if (i < nstarts) st_[i] = starts(i);
        }
    } else {
        // Dynamic solution
        for (int i = 0; i < st_.size(); i++) {
            if (i < start_axis_ || ofs_.size() == 0) {
                st_[i] = 0;
            } else if (i - start_axis_ < (int)ofs_.size()) {
                st_[i] = ofs_[i - start_axis_];
            } else {
                st_[i] = ofs_.back();
            }
        }
    }

    // Determine the ends
    if (start_axis_ < 0) {
        // Static solution
        int nsizes = GET_ARGS_SIZE(sizes);
        for (int i = 0; i < ed_.size(); i++) {
            ed_[i] = X(0).dim(i);
            if (i < nsizes) {
                auto len = sizes(i);
                if (len > 0) { 
                    ed_[i] = st_[i] + len;
                } else if (len == 0) {
                    keep_[i] = 0;
                    ed_[i] = st_[i] + 1;
                }
            }
        }
    } else {
        // Dynamic solution
        for (int i = 0; i < ed_.size(); i++) {
            ed_[i] = X(0).dim(i);
            if (i >= start_axis_) {
                // From a known shape
                auto* sl = ws()->GetTensor(shape_desc_);
                CHECK_EQ(sl->ndim(), X(0).ndim())
                    << "\nThe cropping is performed on "
                    << sl->ndim() << " dimensions, "
                    << "while the num of dimensions of input is "
                    << X(0).ndim() << ".";
                ed_[i] = st_[i] + sl->dim(i);
            }
        }
    }

    // Check starts and ends
    for (int i = 0; i < st_.size(); i++) {
        CHECK(st_[i] >= 0 && st_[i] < X(0).dim(i))
            << "\nThe cropping starts at the pos "
            << st_[i] << " of axis " << i << ", "
            << "while the dimension of this axis is "
            << X(0).dim(i) << ".";
        CHECK(ed_[i] > 0 && ed_[i] <= X(0).dim(i))
            << "\nThe cropping ends at the pos "
            << ed_[i] << " of axis " << i << ", "
            << "while the dimension of this axis is "
            << X(0).dim(i) << ".";
    }

    // Store for the gradient op
    WSTENSOR_FROM_VEC("starts", st_, int64_t);
    WSTENSOR_FROM_VEC("ends", ed_, int64_t);
}

template <class Context>
void CropOp<Context>::RunOnDevice() {
    Setup();

    // Squeeze the dimensions
    vec64_t Y_dims = X(0).dims(), Y_shape;
    for (int i = 0; i < st_.size(); i++) {
        Y_dims[i] = ed_[i] - st_[i];
        if (keep_[i]) Y_shape.push_back(Y_dims[i]);
    }

    Y(0)->Reshape(Y_shape);

    // Just copy the contents
    if (X(0).dims() == Y_dims) {
        Y(0)->CopyFrom(X(0), ctx());
        return;
    }

    TENSOR_FROM_VEC(X_starts_, st_, int);
    TENSOR_FROM_VEC(X_strides_, X(0).strides(), int);
    TENSOR_FROM_VEC(Y_dims_, Y_dims, int);

    DispatchHelper<TensorTypes
        <bool, int8_t, uint8_t, int, int64_t,
               float16, float, double>
    >::Call(this, X(0));
}

template <class Context> template <typename T>
void CropGradientOp<Context>::RunImpl() {
    auto* dy = X(-1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    // Zero the redundant gradients
    math::Set(Y(0)->count(), cast::to<T>(0.f), dx, ctx());

    // Copy the dY to the right positions
    kernel::CropGrad(
        X(-1).count(),
        Y(0)->ndim(),
        X_strides_.template data<int, Context>(),
        Y_dims_.template data<int, Context>(),
        X_starts_.template data<int, Context>(),
        dy, dx, ctx()
    );
}

template <class Context>
void CropGradientOp<Context>::RunOnDevice() {
    VEC_FROM_WSTENSOR("starts", st_, int64_t);
    VEC_FROM_WSTENSOR("ends", ed_, int64_t);

    auto Y_dims = X(0).dims();
    for (int i = 0; i < st_.size(); i++)
        Y_dims[i] = ed_[i] - st_[i];

    Y(0)->ReshapeLike(X(0));

    // Just copy the contents
    if (X(0).dims() == Y_dims) {
        Y(0)->CopyFrom(X(-1), ctx());
        return;
    }

    TENSOR_FROM_VEC(X_starts_, st_, int);
    TENSOR_FROM_VEC(X_strides_, X(0).strides(), int);
    TENSOR_FROM_VEC(Y_dims_, Y_dims, int);

    DispatchHelper<TensorTypes
        <bool, int8_t, uint8_t, int, int64_t,
               float16, float, double>
    >::Call(this, X(0));
}

DEPLOY_CPU(Crop);
#ifdef WITH_CUDA
DEPLOY_CUDA(Crop);
#endif

DEPLOY_CPU(CropGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(CropGradient);
#endif

OPERATOR_SCHEMA(Crop)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(CropGradient)
     /* X, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Crop, SimpleGradientMaker);

#undef TENSOR_FROM_VEC
#undef WSTENSOR_FROM_VEC
#undef VEC_FROM_WSTENSOR

}  // namespace dragon