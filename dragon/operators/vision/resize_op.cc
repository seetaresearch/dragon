#include "dragon/operators/vision/resize_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void ResizeOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);
  Output("X_sizes")->template CopyFrom<int64_t>(in_dims_);
  Output("Y_sizes")->template CopyFrom<int64_t>(out_dims_);

  // Dispatch kernel according to mode and dimensions.
  if (mode_ == "NEAREST") {
    if (out_dims_.size() == 1 || out_dims_.size() == 2) {
      kernels::ResizeNearest2d(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          out_dims_.size() == 1 ? 1 : in_dims_[3],
          out_dims_[0],
          out_dims_.size() == 1 ? 1 : out_dims_[1],
          data_format(),
          X.template data<T, Context>(),
          Y->Reshape(out_shape_)->template mutable_data<T, Context>(),
          ctx());
    } else if (out_dims_.size() == 3) {
      kernels::ResizeNearest3d(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          in_dims_[3],
          in_dims_[4],
          out_dims_[0],
          out_dims_[1],
          out_dims_[2],
          data_format(),
          X.template data<T, Context>(),
          Y->Reshape(out_shape_)->template mutable_data<T, Context>(),
          ctx());
    } else {
      LOG(FATAL) << "ResizeNearest" << out_dims_.size()
                 << "d is not supported.";
    }
  } else if (mode_ == "LINEAR") {
    if (out_dims_.size() == 1 || out_dims_.size() == 2) {
      kernels::ResizeLinear2d(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          out_dims_.size() == 1 ? 1 : in_dims_[3],
          out_dims_[0],
          out_dims_.size() == 1 ? 1 : out_dims_[1],
          align_corners_ > 0,
          data_format(),
          X.template data<T, Context>(),
          Y->Reshape(out_shape_)->template mutable_data<T, Context>(),
          ctx());
    } else {
      LOG(FATAL) << "ResizeLinear" << out_dims_.size() << "d is not supported.";
    }
  } else {
    LOG(FATAL) << "Unknown interpolation mode: " << mode_;
  }
}

template <class Context>
void ResizeOp<Context>::RunOnDevice() {
  auto& X = Input(0);
  CHECK(X.ndim() >= 3) << "\nExcept 3 or more dimensions.";

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
    in_dims_.pop_back(); // Store dimensions in NCHW order.
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

  DispatchHelper<dtypes::Numerical>::Call(this, X);
}

template <class Context>
template <typename T>
void ResizeGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);

  auto* dy = dY.template data<T, Context>();
  auto* dx = dX->template mutable_data<T, Context>();
  auto* dx_acc = (TypeMeta::Id<T>() == TypeMeta::Id<float>())
      ? (float*)nullptr
      : ctx()->workspace()->template data<float, Context>(dX->count());

  // Accumulate to dX.
  if (mode_ == "NEAREST") {
    if (out_dims_.size() == 1 || out_dims_.size() == 2) {
      kernels::ResizeNearest2dGrad(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          out_dims_.size() == 1 ? 1 : in_dims_[3],
          out_dims_[0],
          out_dims_.size() == 1 ? 1 : out_dims_[1],
          data_format(),
          dy,
          dx_acc != nullptr ? dx_acc : reinterpret_cast<float*>(dx),
          ctx());
    } else if (out_dims_.size() == 3) {
      kernels::ResizeNearest3dGrad(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          in_dims_[3],
          in_dims_[4],
          out_dims_[0],
          out_dims_[1],
          out_dims_[2],
          data_format(),
          dy,
          dx_acc != nullptr ? dx_acc : reinterpret_cast<float*>(dx),
          ctx());
    } else {
      LOG(FATAL) << "ResizeNearest" << out_dims_.size()
                 << "d is not supported.";
    }
  } else if (mode_ == "LINEAR") {
    if (out_dims_.size() == 1 || out_dims_.size() == 2) {
      kernels::ResizeLinear2dGrad(
          in_dims_[0],
          in_dims_[1],
          in_dims_[2],
          out_dims_.size() == 1 ? 1 : in_dims_[3],
          out_dims_[0],
          out_dims_.size() == 1 ? 1 : out_dims_[1],
          align_corners_ > 0,
          data_format(),
          dy,
          dx_acc != nullptr ? dx_acc : reinterpret_cast<float*>(dx),
          ctx());
    } else {
      LOG(FATAL) << "ResizeLinear" << out_dims_.size() << "d is not supported.";
    }
  } else {
    LOG(FATAL) << "Unknown interpolation mode: " << mode_;
  }

  // Convert to dX.
  if (dx_acc != nullptr) {
    math::Cast(dX->count(), dx_acc, dx, ctx());
  }
}

template <class Context>
void ResizeGradientOp<Context>::RunOnDevice() {
  Output(0)->ReshapeLike(Input("X_spec"));
  Input("X_sizes").template CopyTo<int64_t>(in_dims_);
  Input("Y_sizes").template CopyTo<int64_t>(out_dims_);
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Resize);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Resize);
#endif

DEPLOY_CPU_OPERATOR(ResizeGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ResizeGradient);
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
