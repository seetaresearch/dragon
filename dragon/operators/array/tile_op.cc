#include "dragon/operators/array/tile_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void TileOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  int num_repeats;
  repeats(0, &num_repeats);
  auto Y_dims = X.dims();
  for (int i = 0; i < num_repeats; ++i) {
    Y_dims[i] *= repeats(i);
  }

  if (X.dims() == Y_dims) {
    Y->Reshape(Y_dims)->CopyFrom(X, ctx());
    return; // Just copy the contents
  }

  kernel::Tile(
      X.ndim(),
      X.dims().data(),
      X.strides().data(),
      Y_dims.data(),
      X.template data<T, Context>(),
      Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
void TileOp<Context>::RunOnDevice() {
  DispatchHelper<AllTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void TileGradientOp<Context>::DoRunWithType() {
  const T* dy;
  T* dx;
  if (src_ == &nav_) {
    dy = ws()->template data<T, Context>({src_->count()})[0];
  } else {
    dy = src_->template data<T, Context>();
  }
  if (dest_ == &nav_) {
    dx = ws()->template data<T, Context>({dest_->count()})[0];
  } else {
    dx = dest_->template mutable_data<T, Context>();
  }
  kernel::TileGrad(
      dest_->count(0, axis_), dest_->count(axis_), repeat_, dy, dx, ctx());
}

template <class Context>
void TileGradientOp<Context>::RunOnDevice() {
  auto &dY = Input(0), *dX = Output(0);

  // Add the axes
  int num_repeats;
  repeats(0, &num_repeats);
  vector<pair<int, int>> dispatch_axes;
  for (int i = 0; i < dY.ndim() && i < num_repeats; i++) {
    auto repeat = repeats(i);
    if (repeat > 1) {
      dispatch_axes.push_back({repeat, i});
    }
  }
  std::sort(dispatch_axes.begin(), dispatch_axes.end());
  std::reverse(dispatch_axes.begin(), dispatch_axes.end());

  if (dispatch_axes.size() == 0) {
    dX->ReshapeLike(dY)->CopyFrom(dY, ctx());
    return; // Just copy the contents
  }

  // Select the initial source and destination
  src_ = &dY, dest_ = dX;
  if (dispatch_axes.size() % 2 == 0) dest_ = &nav_;

  // Reduce N times along each tiled axis
  for (const auto& task : dispatch_axes) {
    axis_ = task.second, repeat_ = task.first;

    vec64_t X_dims(src_->dims());
    X_dims[axis_] /= repeat_;
    dest_->Reshape(X_dims);

    DispatchHelper<FloatingTensorTypes>::Call(this, dY);
    ctx()->FinishDeviceComputation();

    std::swap(src_, dest_);
    if (dispatch_axes.size() % 2 == 1) {
      if (dest_ == &dY) dest_ = &nav_;
    } else {
      if (dest_ == &dY) dest_ = dX;
    }
  }
}

DEPLOY_CPU(Tile);
#ifdef USE_CUDA
DEPLOY_CUDA(Tile);
#endif

DEPLOY_CPU(TileGradient);
#ifdef USE_CUDA
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

REGISTER_GRADIENT(Tile, SimpleGradientMaker);

} // namespace dragon
