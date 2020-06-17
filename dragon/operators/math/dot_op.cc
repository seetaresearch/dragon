#include "dragon/operators/math/dot_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void DotOp<Context>::DotImpl() {
  CHECK_EQ(Input(0).dim(0), Input(1).dim(0))
      << "\nTensor(" << Input(0).name() << "): " << Input(0).DimString()
      << " can not Dot with Tensor"
      << "(" << Input(1).name() << "): " << Input(1).DimString();

  Output(0)->Reshape({});

  auto* a = Input(0).template data<T, Context>();
  auto* b = Input(1).template data<T, Context>();
  auto* y = Output(0)->template mutable_data<T, Context>();

  T yHost;
  math::Dot(Input(0).count(), a, b, &yHost, ctx());
  ctx()->template Copy<T, Context, CPUContext>(1, y, &yHost);
}

template <class Context>
template <typename T>
void DotOp<Context>::GemmImpl() {
  K1_ = transA_ ? Input(0).dim(0) : Input(0).dim(-1);
  K2_ = transB_ ? Input(1).dim(1) : Input(1).dim(0);
  N_ = transB_ ? Input(1).dim(0) : Input(1).dim(1);
  M_ = Input(0).count() / K1_;

  CHECK_EQ(K1_, K2_) << "\nTensor(" << Input(0).name()
                     << "): " << Input(0).DimString()
                     << " can not Dot with Tensor"
                     << "(" << Input(1).name() << "): " << Input(1).DimString();

  auto out_dims = Input(0).dims();
  if (!transA_) {
    out_dims.pop_back();
  } else {
    out_dims.erase(out_dims.begin());
  }
  out_dims.push_back(N_);
  Output(0)->Reshape(out_dims);

  auto* a = Input(0).template data<T, Context>();
  auto* b = Input(1).template data<T, Context>();
  auto* y = Output(0)->template mutable_data<T, Context>();

  math::Gemm(
      transA_ ? CblasTrans : CblasNoTrans,
      transB_ ? CblasTrans : CblasNoTrans,
      M_,
      N_,
      K1_,
      1.f,
      a,
      b,
      0.f,
      y,
      ctx());
}

template <class Context>
template <typename T>
void DotOp<Context>::GemvImpl() {
  N_ = transA_ ? Input(0).dim(0) : Input(0).dim(-1);
  M_ = Input(0).count() / N_;

  CHECK_EQ(N_, Input(1).dim(0))
      << "\nTensor(" << Input(0).name() << "): " << Input(0).DimString()
      << " can not Dot with Tensor"
      << "(" << Input(1).name() << "): " << Input(1).DimString();

  auto out_dims = Input(0).dims();
  if (!transA_) {
    out_dims.pop_back();
  } else {
    out_dims.erase(out_dims.begin());
  }
  Output(0)->Reshape(out_dims);

  auto* a = Input(0).template data<T, Context>();
  auto* b = Input(1).template data<T, Context>();
  auto* y = Output(0)->template mutable_data<T, Context>();

  math::Gemv(
      transA_ ? CblasTrans : CblasNoTrans,
      transA_ ? N_ : M_,
      transA_ ? M_ : N_,
      1.f,
      a,
      b,
      0.f,
      y,
      ctx());
}

template <class Context>
template <typename T>
void DotOp<Context>::DoRunWithType() {
  if (Input(0).ndim() == 1 && Input(1).ndim() == 1) {
    DotImpl<T>();
  } else if (Input(0).ndim() >= 2 && Input(1).ndim() == 2) {
    GemmImpl<T>();
  } else if (Input(0).ndim() >= 2 && Input(1).ndim() == 1) {
    GemvImpl<T>();
  } else {
    LOG(FATAL) << "\nTensor(" << Input(0).name()
               << "): " << Input(0).DimString() << " can not dot with Tensor"
               << "(" << Input(1).name() << "): " << Input(1).DimString();
  }
}

template <class Context>
void DotOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void DotGradientOp<Context>::DotImpl() {
  CHECK_EQ(Input(0).count(), Input(1).count())
      << "\nTensor(" << Input(0).name() << "): " << Input(0).DimString()
      << " can not Dot with Tensor"
      << "(" << Input(1).name() << "): " << Input(1).DimString();

  auto* a = Input(0).template data<T, Context>();
  auto* b = Input(1).template data<T, Context>();
  auto* dy = Input(-1).template data<T, CPUContext>();

  if (Output(0)->has_name()) {
    auto* da = Output(0)->template mutable_data<T, Context>();
    math::Scale(Output(0)->count(), cast::to<float>(dy[0]), b, da, ctx());
  }

  if (Output(1)->has_name()) {
    auto* db = Output(1)->template mutable_data<T, Context>();
    math::Scale(Output(0)->count(), cast::to<float>(dy[0]), a, db, ctx());
  }
}

template <class Context>
template <typename T>
void DotGradientOp<Context>::GemmImpl() {
  K1_ = transA_ ? Input(0).dim(0) : Input(0).dim(-1);
  K2_ = transB_ ? Input(1).dim(1) : Input(1).dim(0);
  N_ = transB_ ? Input(1).dim(0) : Input(1).dim(1);
  M_ = Input(0).count() / K1_;

  CHECK_EQ(K1_, K2_) << "\nTensor(" << Input(0).name()
                     << "): " << Input(0).DimString()
                     << " can not Dot with Tensor"
                     << "(" << Input(1).name() << "): " << Input(1).DimString();

  auto* a = Input(0).template data<T, Context>();
  auto* b = Input(1).template data<T, Context>();
  auto* dy = Input(-1).template data<T, Context>();

  if (Output(0)->has_name()) {
    auto* da = Output(0)->template mutable_data<T, Context>();
    if (transA_) {
      math::Gemm(
          transB_ ? CblasTrans : CblasNoTrans,
          CblasTrans,
          K1_,
          M_,
          N_,
          1.f,
          b,
          dy,
          0.f,
          da,
          ctx());
    } else {
      math::Gemm(
          CblasNoTrans,
          transB_ ? CblasNoTrans : CblasTrans,
          M_,
          K1_,
          N_,
          1.f,
          dy,
          b,
          0.f,
          da,
          ctx());
    }
  }

  if (Output(1)->has_name()) {
    auto* db = Output(1)->template mutable_data<T, Context>();
    if (transB_) {
      math::Gemm(
          CblasTrans,
          transA_ ? CblasTrans : CblasNoTrans,
          N_,
          K1_,
          M_,
          1.f,
          dy,
          a,
          0.f,
          db,
          ctx());
    } else {
      math::Gemm(
          transA_ ? CblasNoTrans : CblasTrans,
          CblasNoTrans,
          K1_,
          N_,
          M_,
          1.f,
          a,
          dy,
          0.f,
          db,
          ctx());
    }
  }
}

template <class Context>
template <typename T>
void DotGradientOp<Context>::GemvImpl() {
  N_ = transA_ ? Input(0).dim(0) : Input(0).dim(-1);
  M_ = Input(0).count() / N_;

  CHECK_EQ(N_, Input(1).dim(0))
      << "\nTensor(" << Input(0).name() << "): " << Input(0).DimString()
      << " can not Dot with Tensor"
      << "(" << Input(1).name() << "): " << Input(1).DimString();

  auto* a = Input(0).template data<T, Context>();
  auto* b = Input(1).template data<T, Context>();
  auto* dy = Input(-1).template data<T, Context>();

  if (Output(0)->has_name()) {
    auto* da = Output(0)->template mutable_data<T, Context>();
    math::Gemm(
        CblasNoTrans, CblasNoTrans, M_, N_, 1, 1.f, dy, b, 0.f, da, ctx());
  }

  if (Output(1)->has_name()) {
    auto* db = Output(1)->template mutable_data<T, Context>();
    math::Gemv(
        transA_ ? CblasNoTrans : CblasTrans,
        transA_ ? N_ : M_,
        transA_ ? M_ : N_,
        1.f,
        a,
        dy,
        0.f,
        db,
        ctx());
  }
}

template <class Context>
template <typename T>
void DotGradientOp<Context>::DoRunWithType() {
  if (Input(0).ndim() == 1 && Input(1).ndim() == 1) {
    DotImpl<T>();
  } else if (Input(0).ndim() >= 2 && Input(1).ndim() == 2) {
    GemmImpl<T>();
  } else if (Input(0).ndim() >= 2 && Input(1).ndim() == 1) {
    GemvImpl<T>();
  } else {
    LOG(FATAL) << "\nTensor(" << Input(0).name()
               << "): " << Input(0).DimString() << " can not Dot with Tensor"
               << "(" << Input(1).name() << "): " << Input(1).DimString();
  }
}

template <class Context>
void DotGradientOp<Context>::RunOnDevice() {
  Output(0)->ReshapeLike(Input(0));
  Output(1)->ReshapeLike(Input(1));
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(-1));
}

DEPLOY_CPU(Dot);
#ifdef USE_CUDA
DEPLOY_CUDA(Dot);
#endif

DEPLOY_CPU(DotGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(DotGradient);
#endif

OPERATOR_SCHEMA(Dot)
    /* A, B */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(DotGradient)
    /* A, B, dY */
    .NumInputs(3)
    /* dA, dB */
    .NumOutputs(2);

REGISTER_GRADIENT(Dot, GenericGradientMaker);

} // namespace dragon
