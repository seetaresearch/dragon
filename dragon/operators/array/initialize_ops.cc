#include "dragon/operators/array/initialize_ops.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

#define DISPATCH_VIA_DTYPES(name, dtypes) \
  template <class Context>                \
  void name##Op<Context>::RunOnDevice() { \
    InitializeOp<Context>::RunOnDevice(); \
    DispatchHelper<dtypes>::Call(this);   \
  }

template <class Context>
void InitializeOp<Context>::RunOnDevice() {
  if (InputSize() > 0) {
    Output(0)->ReshapeLike(Input(0));
  } else {
    vec64_t out_shape;
    int num_dims;
    dims(0, &num_dims);
    for (int i = 0; i < num_dims; ++i) {
      out_shape.push_back(dims(i));
    }
    Output(0)->Reshape(out_shape);
  }
}

template <class Context>
template <typename T>
void FillOp<Context>::DoRunWithType() {
  auto* Y = Output(0);
  math::Set(
      Y->count(),
      convert::To<T>(value_),
      Y->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void RandomNormalOp<Context>::DoRunWithType() {
  auto* Y = Output(0);
  math::RandomNormal(
      Y->count(), mean_, std_, Y->template mutable_data<T, Context>(), ctx());
}

template <class Context>
template <typename T>
void RandomUniformOp<Context>::DoRunWithType() {
  auto* Y = Output(0);
  math::RandomUniform(
      Y->count(), low_, high_, Y->template mutable_data<T, Context>(), ctx());
}

template <class Context>
template <typename T>
void GlorotNormalOp<Context>::DoRunWithType() {
  auto* Y = Output(0);
  auto fan_in = Y->count() / Y->dim(0);
  auto fan_out = Y->ndim() > 1 ? Y->count() / Y->dim(1) : Y->count();
  float fan_num = fan_in;
  if (mode_ == "fan_avg") {
    fan_num = (fan_in + fan_out) * 0.5f;
  } else if (mode_ == "fan_out") {
    fan_num = fan_out;
  }
  math::RandomNormal(
      Y->count(),
      0.f,
      std::sqrt(scale_ / fan_num),
      Y->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void GlorotUniformOp<Context>::DoRunWithType() {
  auto* Y = Output(0);
  auto fan_in = Y->count() / Y->dim(0);
  auto fan_out = Y->ndim() > 1 ? Y->count() / Y->dim(1) : Y->count();
  float fan_num = fan_in;
  if (mode_ == "fan_avg") {
    fan_num = (fan_in + fan_out) * 0.5f;
  } else if (mode_ == "fan_out") {
    fan_num = fan_out;
  }
  math::RandomUniform(
      Y->count(),
      -std::sqrt(scale_ / fan_num),
      std::sqrt(scale_ / fan_num),
      Y->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void TruncatedNormalOp<Context>::DoRunWithType() {
  auto* Y = Output(0);
  CPUContext context;
  math::TruncatedNormal(
      Y->count(),
      mean_,
      std_,
      low_,
      high_,
      Y->template mutable_data<T, CPUContext>(),
      &context);
  Y->template data<T, Context>();
}

template <class Context>
template <typename T>
void EyeOp<Context>::DoRunWithType() {
  auto* Y = Output(0);
  kernels::SetEye(
      Y->count(0, Y->ndim() - 2),
      Y->dim(-2),
      Y->dim(-1),
      k_,
      Y->template mutable_data<T, Context>(),
      ctx());
}

template <class Context>
template <typename T>
void GivenTensorFillOp<Context>::DoRunWithType() {
  Extract<T>();
  CHECK_EQ(Output(0)->count(), values_.count())
      << "\nExcepted the size of output is " << values_.count()
      << ", while got " << Output(0)->count();
  auto* x = values_.template data<T, CPUContext>();
  auto* y = Output(0)->template mutable_data<T, Context>();
  ctx()->template Copy<T, Context, CPUContext>(values_.count(), y, x);
}

template <class Context>
void GivenTensorFillOp<Context>::RunOnDevice() {
  Output(0)->Reshape(shape_);
  DispatchHelper<
      dtypes::TypesBase<int, int64_t, float16, float, double, string>>::
      Call(this, Tensor(dtypes::to_meta(dtype())));
}

DISPATCH_VIA_DTYPES(RandomNormal, dtypes::Floating);
DISPATCH_VIA_DTYPES(RandomUniform, dtypes::Floating);
DISPATCH_VIA_DTYPES(TruncatedNormal, dtypes::Floating);
DISPATCH_VIA_DTYPES(GlorotNormal, dtypes::Floating);
DISPATCH_VIA_DTYPES(GlorotUniform, dtypes::Floating);
DISPATCH_VIA_DTYPES(Fill, dtypes::Generic);
DISPATCH_VIA_DTYPES(Eye, dtypes::Generic);
#undef DISPATCH_VIA_DTYPES

DEPLOY_CPU_OPERATOR(Fill);
DEPLOY_CPU_OPERATOR(Eye);
DEPLOY_CPU_OPERATOR(GivenTensorFill);
DEPLOY_CPU_OPERATOR(RandomNormal);
DEPLOY_CPU_OPERATOR(RandomUniform);
DEPLOY_CPU_OPERATOR(GlorotNormal);
DEPLOY_CPU_OPERATOR(GlorotUniform);

#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Fill);
DEPLOY_CUDA_OPERATOR(Eye);
DEPLOY_CUDA_OPERATOR(GivenTensorFill);
DEPLOY_CUDA_OPERATOR(RandomNormal);
DEPLOY_CUDA_OPERATOR(RandomUniform);
DEPLOY_CUDA_OPERATOR(GlorotNormal);
DEPLOY_CUDA_OPERATOR(GlorotUniform);
DEPLOY_CPU_CUDA_OPERATOR(TruncatedNormal);
#else
DEPLOY_CPU_OPERATOR(TruncatedNormal);
#endif

OPERATOR_SCHEMA(Fill).NumInputs(0, 1).NumOutputs(1);
OPERATOR_SCHEMA(Eye).NumInputs(0, 1).NumOutputs(1);
OPERATOR_SCHEMA(GivenTensorFill).NumInputs(0).NumOutputs(1);
OPERATOR_SCHEMA(RandomUniform).NumInputs(0, 1).NumOutputs(1);
OPERATOR_SCHEMA(RandomNormal).NumInputs(0, 1).NumOutputs(1);
OPERATOR_SCHEMA(GlorotUniform).NumInputs(0, 1).NumOutputs(1);
OPERATOR_SCHEMA(GlorotNormal).NumInputs(0, 1).NumOutputs(1);
OPERATOR_SCHEMA(TruncatedNormal).NumInputs(0, 1).NumOutputs(1);

NO_GRADIENT(Fill);
NO_GRADIENT(Eye);
NO_GRADIENT(GivenTensorFill);
NO_GRADIENT(RandomUniform);
NO_GRADIENT(RandomNormal);
NO_GRADIENT(GlorotUniform);
NO_GRADIENT(GlorotNormal);
NO_GRADIENT(TruncatedNormal);

} // namespace dragon
