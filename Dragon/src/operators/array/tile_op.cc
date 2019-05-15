#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "operators/array/tile_op.h"

namespace dragon {

#define PUT_X_AND_Y(x, y) \
    const T* x; T* y; \
    if (src_ == &nav_) { \
        x = ws()->template data<T, Context> \
                ({ src_->count() })[0]; \
    } else { \
        x = src_->template data<T, Context>(); \
    } \
    if (dst_ == &nav_) { \
        y = ws()->template data<T, Context> \
                ({ dst_->count() })[0]; \
    } else { \
        y = dst_->template mutable_data<T, Context>(); \
    }

#define TENSOR_FROM_VEC(tensor, vec, T) \
    { \
        tensor.Reshape({ (int64_t)vec.size() }); \
        auto* data = tensor.template mutable_data<T, CPUContext>(); \
        for (int i = 0; i < vec.size(); i++) data[i] = (T)vec[i]; \
    }

template <class Context> template <typename T>
void TileOp<Context>::RunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    // Apply a simple Nd-Broadcast solution
    kernel::Tile(
        Y(0)->count(),
        Y(0)->ndim(),
        X_dims_.template data<int, Context>(),
        X_strides_.template data<int, Context>(),
        X_dims_.template data<int, Context>(),
        x, y, ctx()
    );
}

template <class Context>
void TileOp<Context>::RunOnDevice() {
    auto Y_dims = X(0).dims();
    for (int i = 0; i < Y_dims.size(); ++i)
        Y_dims[i] *= multiples(i);
    
    Y(0)->Reshape(Y_dims);

    // Just copy the contents
    if (X(0).dims() == Y_dims) {
        Y(0)->CopyFrom(X(0), ctx());
        return;
    }

    TENSOR_FROM_VEC(X_strides_, X(0).strides(), int);
    TENSOR_FROM_VEC(X_dims_, X(0).dims(), int);
    TENSOR_FROM_VEC(Y_dims_, Y_dims, int);

    DispatchHelper<TensorTypes
        <bool, int8_t, uint8_t, int, int64_t,
               float16, float, double>
    >::Call(this, X(0));
}

template <class Context> template <typename T>
void TileGradientOp<Context>::RunImpl() {
    PUT_X_AND_Y(dy, dx);

    kernel::TileGrad(
        rows_,
        cols_,
        multiple_,
        dy, dx, ctx()
    );
}

template <class Context>
void TileGradientOp<Context>::RunOnDevice() {
    // Add the axes
    vector<pair<int, int>> dispatch_axes;
    for (int i = 0; i < X(0).ndim(); i++) {
        auto m = multiples(i);
        if (m > 1) { dispatch_axes.push_back({ m, i }); }
    }
    std::sort(dispatch_axes.begin(), dispatch_axes.end());
    std::reverse(dispatch_axes.begin(), dispatch_axes.end());

    // Just copy the contents
    if (dispatch_axes.size() == 0) {
        Y(0)->ReshapeLike(X(-1));
        Y(0)->CopyFrom(X(-1), ctx());
        return;
    }

    // Select the src && dst
    src_ = &X(0); dst_ = Y(0);
    if (dispatch_axes.size() % 2 == 0) dst_ = &nav_;

    for (const auto& task : dispatch_axes) {
        axis_ = task.second;
        multiple_ = task.first;

        auto out_shape = src_->dims();
        out_shape[axis_] /= multiple_;
        dst_->Reshape(out_shape);
        rows_ = dst_->count(0, axis_);
        cols_ = dst_->count(axis_);

        DispatchHelper<TensorTypes
            <int8_t, uint8_t, int, int64_t,
                float16, float, double>
        >::Call(this, X(0));
        ctx()->FinishDeviceCompution();

        // Protect X if num_axes >= 2
        std::swap(src_, dst_);
        if (dispatch_axes.size() % 2 == 1) {
            if (dst_ == &X(-1)) dst_ = &nav_;
        } else {
            if (dst_ == &X(-1)) dst_ = Y(0);
        }
    }
}

DEPLOY_CPU(Tile);
#ifdef WITH_CUDA
DEPLOY_CUDA(Tile);
#endif

DEPLOY_CPU(TileGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(TileGradient);
#endif

OPERATOR_SCHEMA(Tile)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(TileGradient)
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

REGISTER_GRADIENT(Tile, GradientMaker);

#undef PUT_X_AND_Y
#undef TENSOR_FROM_VEC

}  // namespace dragon