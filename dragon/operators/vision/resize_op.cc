#include "dragon/operators/vision/resize_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void ResizeOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  // Store for the gradient calculation
  STORE_INPUT_SPEC(0);
  Buffer("in_dims")->template CopyFrom<int64_t>(in_dims_);
  Buffer("out_dims")->template CopyFrom<int64_t>(out_dims_);

  if (out_dims_.size() == 2) {
    if (mode_ == "NEAREST") {
      kernel::ResizeNearest(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          in_dims_[3],
          out_dims_[0],
          out_dims_[1],
          data_format(),
          X.template data<T, Context>(),
          Y->Reshape(out_shape_)->template mutable_data<T, Context>(),
          ctx());
    } else if (mode_ == "LINEAR") {
      kernel::ResizeLinear(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          in_dims_[3],
          out_dims_[0],
          out_dims_[1],
          align_corners_ > 0,
          data_format(),
          X.template data<T, Context>(),
          Y->Reshape(out_shape_)->template mutable_data<T, Context>(),
          ctx());
    } else {
      LOG(FATAL) << "Unknown interpolation mode: " << mode_;
    }
  } else {
    LOG(FATAL) << "Resize" << out_dims_.size() << "d is not supported.";
  }
}

template <class Context>
void ResizeOp<Context>::RunOnDevice() {
  auto& X = Input(0);
  CHECK(X.ndim() >= 3 && X.ndim() <= 5)
      << "\nOnly 3d/4d/5d input are supported.";

  int axis = 1;
  int num_axes = X.ndim() - 2;
  int num_sizes;
  sizes(0, &num_sizes);
  int num_scales;
  scales(0, &num_scales);

  in_dims_ = X.dims();
  if (data_format() == "NCHW") {
    axis = num_axes;
  } else if (data_format() == "NHWC") {
    in_dims_.insert(in_dims_.begin() + 1, in_dims_.back());
    in_dims_.pop_back(); // Store dimensions in NCHW order
  } else {
    LOG(FATAL) << "Unknown data format: " << data_format();
  }

  out_shape_ = X.dims();
  out_dims_.resize((size_t)num_axes);

  if (num_sizes > 0) {
    if (num_sizes == 1) {
      for (int i = 0; i < num_axes; ++i)
        out_dims_[i] = out_shape_[axis + i] = sizes(0);
    } else if (num_sizes == num_axes) {
      for (int i = 0; i < num_axes; ++i)
        out_dims_[i] = out_shape_[axis + i] = sizes(i);
    } else {
      CHECK_EQ(num_sizes, X.ndim())
          << "\nExcepted 1/" << num_axes << "/" << X.ndim() << " values "
          << "for <sizes>, got " << num_sizes << ".";
      for (int i = 0; i < num_axes; ++i)
        out_dims_[i] = out_shape_[axis + i] = sizes(axis + i);
    }
  } else if (num_scales > 0) {
    if (num_scales == 1) {
      for (int i = 0; i < num_axes; ++i) {
        out_shape_[axis + i] *= scales(0);
        out_dims_[i] = out_shape_[axis + i];
      }
    } else if (num_scales == num_axes) {
      for (int i = 0; i < num_axes; ++i) {
        out_shape_[axis + i] *= scales(i);
        out_dims_[i] = out_shape_[axis + i];
      }
    } else {
      CHECK_EQ(num_scales, X.ndim())
          << "\nExcepted 1/" << num_axes << "/" << X.ndim() << " values "
          << "for <scales>, got " << num_scales << ".";
      for (int i = 0; i < num_axes; ++i) {
        out_shape_[axis + i] *= scales(axis + i);
        out_dims_[i] = out_shape_[axis + i];
      }
    }
  } else {
    LOG(FATAL) << "Specify either <sizes> or <scales>.";
  }

  DispatchHelper<MathTensorTypes>::Call(this, X);
}

template <class Context>
template <typename Ty, typename Tx>
void ResizeGradientOp<Context>::NearestImpl(const Ty* dy, Tx* dx) {
  if (out_dims_.size() == 2) {
    kernel::ResizeNearestGrad(
        in_dims_[0],
        in_dims_[1],
        in_dims_[2],
        in_dims_[3],
        out_dims_[0],
        out_dims_[1],
        data_format(),
        dy,
        dx,
        ctx());
  } else {
    LOG(FATAL) << "Resize" << out_dims_.size() << "d is not supported.";
  }
}

template <class Context>
template <typename Ty, typename Tx>
void ResizeGradientOp<Context>::LinearImpl(const Ty* dy, Tx* dx) {
  if (out_dims_.size() == 2) {
    kernel::ResizeLinearGrad(
        in_dims_[0],
        in_dims_[1],
        in_dims_[2],
        in_dims_[3],
        out_dims_[0],
        out_dims_[1],
        align_corners_ > 0,
        data_format(),
        dy,
        dx,
        ctx());
  } else {
    LOG(FATAL) << "Resize" << out_dims_.size() << "d is not supported.";
  }
}

template <class Context>
template <typename T>
void ResizeGradientOp<Context>::DoRunWithType() {
  auto* dy = Input(0).template data<T, Context>();
  auto* dx = Output(0)->template mutable_data<T, Context>();
  if (mode_ == "NEAREST") {
    NearestImpl(dy, dx);
  } else if (mode_ == "LINEAR") {
    LinearImpl(dy, dx);
  } else {
    LOG(FATAL) << "Unknown interpolation mode: " << mode_;
  }
}

template <class Context>
template <typename T>
void ResizeGradientOp<Context>::DoRunWithTypeAndCast() {
  auto* dy = Input(0).template data<T, Context>();
  auto* dx = Output(0)->template mutable_data<T, Context>();
  auto* scratch = ws()->template data<float, Context>({Output(0)->count()})[0];
  if (mode_ == "NEAREST") {
    NearestImpl(dy, scratch);
  } else if (mode_ == "LINEAR") {
    LinearImpl(dy, scratch);
  } else {
    LOG(FATAL) << "Unknown interpolation mode: " << mode_;
  }
  kernel::Cast(Output(0)->count(), scratch, dx, ctx());
}

template <class Context>
void ResizeGradientOp<Context>::RunOnDevice() {
  Output(0)->ReshapeLike(RESTORE_INPUT_SPEC(0));
  Buffer("in_dims")->template CopyTo<int64_t>(in_dims_);
  Buffer("out_dims")->template CopyTo<int64_t>(out_dims_);

  if (XIsType(Input(0), float16)) {
    DoRunWithTypeAndCast<float16>();
  } else if (XIsType(Input(0), float)) {
    DoRunWithType<float>();
  } else if (XIsType(Input(0), double)) {
    DoRunWithTypeAndCast<double>();
  } else {
    LOG(FATAL) << TypeString(Input(0), {"float16", "float32", "float64"});
  };
}

DEPLOY_CPU(Resize);
#ifdef USE_CUDA
DEPLOY_CUDA(Resize);
#endif

DEPLOY_CPU(ResizeGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(ResizeGradient);
#endif

OPERATOR_SCHEMA(Resize)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(ResizeGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Resize, SimpleGradientMaker);

} // namespace dragon
