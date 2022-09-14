#include "dragon/operators/math/reduce_op.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {

template <class Context, class Reducer>
template <typename T>
void ReduceOp<Context, Reducer>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  // Compute reduce axes.
  vec64_t Y_dims(X.dims()), Y_shape(X.dims());
  vec64_t reduce_axes(axes_.begin(), axes_.end());
  if (axes_.empty()) {
    reduce_axes.resize(X.ndim());
    for (int i = 0; i < X.ndim(); ++i) {
      reduce_axes[i] = i;
    }
  }
  for (int i = 0; i < reduce_axes.size(); ++i) {
    auto axis = reduce_axes[i];
    reduce_axes[i] = axis = axis < 0 ? axis + X.ndim() : axis;
    CHECK(axis >= 0 && axis < X.ndim())
        << "\nExcepted the axis in [-" << X.ndim() << ", " << X.ndim()
        << "), got " << axis << ".";
    Y_dims[axis] = 1;
    Y_shape[axis] = keep_dims_ ? 1 : -1;
  }

  // Squeeze output shape.
  const auto& erase_iter = std::remove_if(
      Y_shape.begin(), Y_shape.end(), [](int64_t x) { return x == -1; });
  Y_shape.erase(erase_iter, Y_shape.end());

  // Save context for gradient computation.
  Output("X_spec")->ReshapeLike(X);
  Output("Y_dims")->template CopyFrom<int64_t>(Y_dims);

  // Reduce or Copy X to Y.
  Y->Reshape(Y_shape);
  if (X.count() == 1 || X.count() == Y->count()) {
    Y->CopyFrom(X, ctx());
  } else {
    reducer_.template Compute<T, Context>(
        X.dims(),
        reduce_axes,
        X.template data<T, Context>(),
        Y->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
template <typename T>
void ReduceSumGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0)->ReshapeLike(Input("X_spec"));

  // Broadcast or Copy dY to dX.
  if (dX->count() == 1 || dX->count() == dY.count()) {
    dX->CopyFrom(dY, ctx());
  } else {
    vec64_t Y_dims;
    Input("Y_dims").template CopyTo<int64_t>(Y_dims);
    kernels::ReduceSumGrad(
        dX->ndim(),
        dX->dims().data(),
        Y_dims.data(),
        Tensor(Y_dims).strides().data(),
        1.f,
        dY.template data<T, Context>(),
        dX->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
template <typename T>
void ReduceMeanGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0)->ReshapeLike(Input("X_spec"));

  // Broadcast or Copy dY to dX.
  if (dX->count() == 1 || dX->count() == dY.count()) {
    dX->CopyFrom(dY, ctx());
  } else {
    vec64_t Y_dims;
    Input("Y_dims").template CopyTo<int64_t>(Y_dims);
    kernels::ReduceSumGrad(
        dX->ndim(),
        dX->dims().data(),
        Y_dims.data(),
        Tensor(Y_dims).strides().data(),
        1.f / float(dX->count() / dY.count()),
        dY.template data<T, Context>(),
        dX->template mutable_data<T, Context>(),
        ctx());
  }
}

DEPLOY_CPU_OPERATOR(ReduceMax);
DEPLOY_CPU_OPERATOR(ReduceMin);
DEPLOY_CPU_OPERATOR(ReduceSum);
DEPLOY_CPU_OPERATOR(ReduceMean);
DEPLOY_CPU_OPERATOR(ReduceVar);
DEPLOY_CPU_OPERATOR(ReduceL1);
DEPLOY_CPU_OPERATOR(ReduceL2);
DEPLOY_CPU_OPERATOR(ReduceSumGradient);
DEPLOY_CPU_OPERATOR(ReduceMeanGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(ReduceMax);
DEPLOY_CUDA_OPERATOR(ReduceMin);
DEPLOY_CUDA_OPERATOR(ReduceSum);
DEPLOY_CUDA_OPERATOR(ReduceMean);
DEPLOY_CUDA_OPERATOR(ReduceVar);
DEPLOY_CUDA_OPERATOR(ReduceL1);
DEPLOY_CUDA_OPERATOR(ReduceL2);
DEPLOY_CUDA_OPERATOR(ReduceSumGradient);
DEPLOY_CUDA_OPERATOR(ReduceMeanGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(ReduceMax, ReduceMax);
DEPLOY_MPS_OPERATOR(ReduceMin, ReduceMin);
DEPLOY_MPS_OPERATOR(ReduceSum, ReduceSum);
DEPLOY_MPS_OPERATOR(ReduceMean, ReduceMean);
DEPLOY_MPS_OPERATOR(ReduceVar, ReduceVar);
DEPLOY_MPS_OPERATOR(ReduceSumGradient, ReduceSumGradient);
DEPLOY_MPS_OPERATOR(ReduceMeanGradient, ReduceMeanGradient);
#endif

/* X -> Y */
OPERATOR_SCHEMA(ReduceMax).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(ReduceMin).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(ReduceSum).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(ReduceMean).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(ReduceVar).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(ReduceL1).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(ReduceL2).NumInputs(1).NumOutputs(1);
/* dY -> dX */
OPERATOR_SCHEMA(ReduceSumGradient).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(ReduceMeanGradient).NumInputs(1).NumOutputs(1);

NO_GRADIENT(ReduceMax);
NO_GRADIENT(ReduceMin);
NO_GRADIENT(ReduceVar);
NO_GRADIENT(ReduceL1);
NO_GRADIENT(ReduceL2);
REGISTER_GRADIENT(ReduceSum, SimpleGradientMaker);
REGISTER_GRADIENT(ReduceMean, SimpleGradientMaker);

} // namespace dragon
