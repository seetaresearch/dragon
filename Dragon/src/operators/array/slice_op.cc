#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/array/slice_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGS(Input) \
    axis_ = OpArg<int64_t>("axis", 0); \
    axis_ = axis_ < 0 ? axis_ + Input.ndim() : axis_; \
    CHECK(axis_ >= 0 && axis_ < Input.ndim()) \
       << "\nExcepted the axis in [-" << Input.ndim() \
       << ", " << Input.ndim() << "), got " \
       << OpArg<int64_t>("axis", 0) << "."; \
    if (points_.empty()) { \
        CHECK_EQ(X(0).dim(axis_) % N_, 0) \
            << "\nSelected dim is " << X(0).dim(axis_) \
            << ", can't be sliced into " << N_ << " parts."; \
    }

template <class Context> template <typename T>
void SliceOp<Context>::RunImpl() {
    int64_t slice_ofs = 0;
    auto out_shape = X(0).dims();
    auto* x = X(0).template data<T, Context>();

    for (int i = 0; i < N_; i++) {
        if (!points_.empty()) {
            slice_dim_ = i < N_ - 1 ?
                points_[i] - slice_ofs :
                    axis_dim_ - slice_ofs;
        }

        CHECK(slice_dim_ > 0 &&
              slice_ofs + slice_dim_ <= axis_dim_)
          << "\nIllegal slicing points: "
          << Tensor::DimString(points_)
          << " for dimension: " << axis_dim_;

        out_shape[axis_] = slice_dim_;
        Y(i)->Reshape(out_shape);

        auto* y = Y(i)->template
            mutable_data<T, Context>();

        kernel::Slice(
            outer_dim_,
            inner_dim_,
            axis_dim_,
            slice_dim_,
            slice_ofs,
            x, y, ctx()
        );

        slice_ofs += slice_dim_;
    }
}

template <class Context>
void SliceOp<Context>::RunOnDevice() {
    N_ = YSize();  /* num_slices */
    DETERMINE_RUNTIME_ARGS(X(0));

    axis_dim_  = X(0).dim(axis_);
    slice_dim_ = axis_dim_ / N_;
    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);

    DispatchHelper<TensorTypes
        <bool, int8_t, uint8_t, int, int64_t,
               float16, float, double>
    >::Call(this, X(0));
}

template <class Context> template <typename T>
void SliceGradientOp<Context>::RunImpl() {
    int64_t slice_ofs = 0;
    auto* dx = Y(0)->template mutable_data<T, Context>();

    for (int i = 0; i < N_; i++) {
        if (!points_.empty()) {
            slice_dim_ = i < N_ - 1 ?
                points_[i] - slice_ofs :
                    axis_dim_ - slice_ofs;
        }

        CHECK(slice_dim_ > 0 &&
              slice_ofs + slice_dim_ <= axis_dim_)
            << "\nIllegal slicing points: "
            << Tensor::DimString(points_)
            << " for dimension: " << axis_dim_;

        if (X(i + 1).name() != "NULL") {
            auto* dy = X(i + 1)
                .template data<T, Context>();
            kernel::SliceGrad(
                outer_dim_,
                inner_dim_,
                axis_dim_,
                slice_dim_,
                slice_ofs,
                dy, dx, ctx()
            );
        }

        slice_ofs += slice_dim_;
    }
}

template <class Context>
void SliceGradientOp<Context>::RunOnDevice() {
    N_ = XSize() - 1;  /* num_slices */
    DETERMINE_RUNTIME_ARGS(X(0));

    axis_dim_  = X(0).dim(axis_);
    slice_dim_ = axis_dim_ / N_;
    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);

    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <bool, int8_t, uint8_t, int, int64_t,
               float16, float, double>
    >::Call(this, X(0));
}

DEPLOY_CPU(Slice);
#ifdef WITH_CUDA
DEPLOY_CUDA(Slice);
#endif

DEPLOY_CPU(SliceGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SliceGradient);
#endif

OPERATOR_SCHEMA(Slice)
     /* X */
    .NumInputs(1)
     /* Y(0), ... */
    .NumOutputs(1, INT_MAX);

OPERATOR_SCHEMA(SliceGradient)
     /* X, dY(0), ... */
    .NumInputs(2, INT_MAX)
     /* dX */
    .NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        vector<string> inputs({ I(0) });
        for (int i = 0; i < def.output_size(); i++)
            inputs.push_back(GO(i));
        return SingleDef(
            def.type() + "Gradient", "",
            inputs, vector<string>({ GI(0) })
        );
    }
};

}  // namespace

REGISTER_GRADIENT(Slice, GradientMaker);

#undef DETERMINE_RUNTIME_ARGS

}  // namespace dragon