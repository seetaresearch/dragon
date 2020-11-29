#include "dragon/operators/array/initialize_ops.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

#define DEFINE_FILLER_OP_IMPL(name)                        \
  template <class Context>                                 \
  template <typename T>                                    \
  void name##Op<Context>::DoRunWithType() {                \
    unique_ptr<Filler<T, Context>> f;                      \
    f.reset(CreateFiller<T, Context>(this->filler_info_)); \
    f->Fill(Output(0), ctx());                             \
  }

#define DISPATCH_WITH_TYPES(name, ...)                    \
  template <class Context>                                \
  void name##Op<Context>::RunOnDevice() {                 \
    InitializeOp<Context>::RunOnDevice();                 \
    DispatchHelper<TensorTypes<__VA_ARGS__>>::Call(this); \
  }

#define DISPATCH_WITH_TENSOR_TYPES(name, tensor_types) \
  template <class Context>                             \
  void name##Op<Context>::RunOnDevice() {              \
    InitializeOp<Context>::RunOnDevice();              \
    DispatchHelper<tensor_types>::Call(this);          \
  }

template <class Context>
void InitializeOp<Context>::RunOnDevice() {
  if (InputSize() > 0) {
    Output(0)->ReshapeLike(Input(0));
  } else {
    vec64_t out_shape;
    int ndims;
    dims(0, &ndims);
    for (int i = 0; i < ndims; i++) {
      out_shape.push_back(dims(i));
    }
    Output(0)->Reshape(out_shape);
  }
}

template <class Context>
template <typename T>
void FillOp<Context>::DoRunWithType() {
  auto* y = Output(0)->template mutable_data<T, Context>();
  math::Set(Output(0)->count(), convert::To<T>(value_), y, ctx());
}

template <class Context>
template <typename T>
void EyeOp<Context>::DoRunWithType() {
  auto* y = Output(0)->template mutable_data<T, Context>();
  kernel::Eye(Output(0)->dim(0), Output(0)->dim(1), k_, y, ctx());
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

  DispatchHelper<TensorTypes<int, int64_t, float16, float, double, string>>::
      Call(this, Tensor(types::to_meta(dtype())));
}

DEFINE_FILLER_OP_IMPL(RandomNormal);
DEFINE_FILLER_OP_IMPL(RandomUniform);
DEFINE_FILLER_OP_IMPL(TruncatedNormal);
DEFINE_FILLER_OP_IMPL(GlorotNormal);
DEFINE_FILLER_OP_IMPL(GlorotUniform);
#undef DEFINE_FILLER_OP_IMPL

DISPATCH_WITH_TENSOR_TYPES(RandomNormal, FloatingTensorTypes);
DISPATCH_WITH_TENSOR_TYPES(RandomUniform, FloatingTensorTypes);
DISPATCH_WITH_TENSOR_TYPES(TruncatedNormal, FloatingTensorTypes);
DISPATCH_WITH_TENSOR_TYPES(GlorotNormal, FloatingTensorTypes);
DISPATCH_WITH_TENSOR_TYPES(GlorotUniform, FloatingTensorTypes);
DISPATCH_WITH_TENSOR_TYPES(Fill, FullTensorTypes);
DISPATCH_WITH_TENSOR_TYPES(Eye, FullTensorTypes);
#undef DISPATCH_WITH_TYPES
#undef DISPATCH_WITH_TENSOR_TYPES

DEPLOY_CPU_OPERATOR(Fill);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Fill);
#endif

DEPLOY_CPU_OPERATOR(Eye);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Eye);
#endif

DEPLOY_CPU_OPERATOR(GivenTensorFill);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(GivenTensorFill);
#endif

DEPLOY_CPU_OPERATOR(RandomNormal);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(RandomNormal);
#endif

DEPLOY_CPU_OPERATOR(RandomUniform);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(RandomUniform);
#endif

#ifdef USE_CUDA
DEPLOY_CPU_CUDA_OPERATOR(TruncatedNormal);
#else
DEPLOY_CPU_OPERATOR(TruncatedNormal);
#endif

DEPLOY_CPU_OPERATOR(GlorotNormal);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(GlorotNormal);
#endif

DEPLOY_CPU_OPERATOR(GlorotUniform);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(GlorotUniform);
#endif

OPERATOR_SCHEMA(Fill).NumInputs(0, 1).NumOutputs(1);
OPERATOR_SCHEMA(Eye).NumInputs(0, 1).NumOutputs(1);
OPERATOR_SCHEMA(GivenTensorFill).NumInputs(0).NumOutputs(1);
OPERATOR_SCHEMA(RandomUniform).NumInputs(0, 1).NumOutputs(1);
OPERATOR_SCHEMA(RandomNormal).NumInputs(0, 1).NumOutputs(1);
OPERATOR_SCHEMA(TruncatedNormal).NumInputs(0, 1).NumOutputs(1);
OPERATOR_SCHEMA(GlorotUniform).NumInputs(0, 1).NumOutputs(1);
OPERATOR_SCHEMA(GlorotNormal).NumInputs(0, 1).NumOutputs(1);

NO_GRADIENT(Fill);
NO_GRADIENT(Eye);
NO_GRADIENT(GivenTensorFill);
NO_GRADIENT(RandomUniform);
NO_GRADIENT(RandomNormal);
NO_GRADIENT(TruncatedNormal);
NO_GRADIENT(GlorotUniform);
NO_GRADIENT(GlorotNormal);

} // namespace dragon
