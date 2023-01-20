/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_MATH_REDUCE_OP_H_
#define DRAGON_OPERATORS_MATH_REDUCE_OP_H_

#include "dragon/core/operator.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/operators/math/reduce_op_impl_cnnl.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context, class Reducer>
class ReduceOp : public Operator<Context> {
 public:
  ReduceOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        axes_(OP_REPEATED_ARG(int64_t, "axes")),
        keep_dims_(OP_SINGLE_ARG(int64_t, "keepdims", 0)) {}
  USE_OPERATOR_FUNCTIONS;

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t keep_dims_;
  vec64_t axes_;
  Reducer reducer_;
};

namespace {

struct VarReducer {
  template <typename T, class Context>
  inline void Compute(
      const vec64_t& dims,
      const vec64_t& axes,
      const T* x,
      T* y,
      Context* ctx) const {
    kernels::Moments(
        dims.size(),
        dims.data(),
        axes.size(),
        axes.data(),
        x,
        y, // Placeholder.
        y,
        ctx);
  }
};

struct L2Reducer {
  template <typename T, class Context>
  inline void Compute(
      const vec64_t& dims,
      const vec64_t& axes,
      const T* x,
      T* y,
      Context* ctx) const {
    int64_t N = math::utils::Prod(dims);
    for (const auto axis : axes) {
      N /= dims[axis];
    }
    math::ReduceSumSqr(
        dims.size(), dims.data(), axes.size(), axes.data(), 1.f, x, y, ctx);
    math::Sqrt(N, y, y, ctx);
  }
};

#define DEFINE_REDUCER(name, ReduceFunc, kScale) \
  struct name {                                  \
    template <typename T, class Context>         \
    inline void Compute(                         \
        const vec64_t& dims,                     \
        const vec64_t& axes,                     \
        const T* x,                              \
        T* y,                                    \
        Context* ctx) const {                    \
      int64_t C = 1;                             \
      for (const auto axis : axes) {             \
        C *= dims[axis];                         \
      }                                          \
      return ReduceFunc(                         \
          dims.size(),                           \
          dims.data(),                           \
          axes.size(),                           \
          axes.data(),                           \
          kScale,                                \
          x,                                     \
          y,                                     \
          ctx);                                  \
    }                                            \
  };

DEFINE_REDUCER(MaxReducer, math::ReduceMax, 1.f);
DEFINE_REDUCER(MinReducer, math::ReduceMin, 1.f);
DEFINE_REDUCER(SumReducer, math::ReduceSum, 1.f);
DEFINE_REDUCER(MeanReducer, math::ReduceSum, 1.f / C);
DEFINE_REDUCER(L1Reducer, math::ReduceL1, 1.f);
#undef DEFINE_REDUCER

} // namespace

#define DEFINE_REDUCE_OP(name, BaseOp, Reducer, ReduceTypes) \
  template <class Context>                                   \
  class name##Op final : public BaseOp<Context, Reducer> {   \
   public:                                                   \
    name##Op(const OperatorDef& def, Workspace* ws)          \
        : BaseOp<Context, Reducer>(def, ws) {}               \
    USE_OPERATOR_FUNCTIONS;                                  \
                                                             \
    void RunOnDevice() override {                            \
      DispatchHelper<ReduceTypes>::Call(this, Input(0));     \
    }                                                        \
  };

DEFINE_REDUCE_OP(ReduceMax, ReduceOp, MaxReducer, dtypes::Numerical);
DEFINE_REDUCE_OP(ReduceMin, ReduceOp, MinReducer, dtypes::Numerical);
DEFINE_REDUCE_OP(ReduceSum, ReduceOp, SumReducer, dtypes::Accumulated);
DEFINE_REDUCE_OP(ReduceMean, ReduceOp, MeanReducer, dtypes::Accumulated);
DEFINE_REDUCE_OP(ReduceVar, ReduceOp, VarReducer, dtypes::Loss);
DEFINE_REDUCE_OP(ReduceL1, ReduceOp, L1Reducer, dtypes::Floating);
DEFINE_REDUCE_OP(ReduceL2, ReduceOp, L2Reducer, dtypes::Floating);

#define DEFINE_REDUCE_GRAD_OP(name)                           \
  template <class Context>                                    \
  class name##GradientOp final : public Operator<Context> {   \
   public:                                                    \
    SIMPLE_CTOR_DTOR(name##GradientOp);                       \
    USE_OPERATOR_FUNCTIONS;                                   \
                                                              \
    void RunOnDevice() override {                             \
      DispatchHelper<dtypes::Floating>::Call(this, Input(0)); \
    }                                                         \
                                                              \
    template <typename T>                                     \
    void DoRunWithType();                                     \
  };

DEFINE_REDUCE_GRAD_OP(ReduceMean);
DEFINE_REDUCE_GRAD_OP(ReduceSum);

#ifdef USE_MLU
#define DEFINE_CNNL_REDUCE_OP(name, Reducer, ReduceTypes) \
  DEFINE_REDUCE_OP(CNNL##name, CNNLReduceOp, Reducer, ReduceTypes)

#define DEFINE_CNNL_REDUCE_GRAD_OP(name)                          \
  template <class Context>                                        \
  class CNNL##name##GradientOp : public Operator<Context> {       \
   public:                                                        \
    CNNL##name##GradientOp(const OperatorDef& def, Workspace* ws) \
        : Operator<Context>(def, ws) {                            \
      CNNLCreateTensorDesc(&input_desc_);                         \
      CNNLCreateTensorDesc(&output_desc_);                        \
    }                                                             \
    USE_OPERATOR_FUNCTIONS;                                       \
    ~CNNL##name##GradientOp() {                                   \
      CNNLDestroyTensorDesc(input_desc_);                         \
      CNNLDestroyTensorDesc(output_desc_);                        \
    }                                                             \
    void RunOnDevice() override {                                 \
      DispatchHelper<dtypes::Floating>::Call(this, Input(0));     \
    }                                                             \
    template <typename T>                                         \
    void DoRunWithType();                                         \
                                                                  \
   protected:                                                     \
    cnnlTensorDescriptor_t input_desc_, output_desc_;             \
  };

template <class Context, cnnlReduceOp_t Reducer>
class CNNLReduceOp : public Operator<Context> {
 public:
  CNNLReduceOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        axes_(OP_REPEATED_ARG(int64_t, "axes")),
        keep_dims_(OP_SINGLE_ARG(int64_t, "keepdims", 0)) {
    impl_.SetReducer(Reducer);
  }
  USE_OPERATOR_FUNCTIONS;

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t keep_dims_;
  vec64_t axes_;
  CNNLReduceOpImpl impl_;
};

DEFINE_CNNL_REDUCE_OP(ReduceMax, CNNL_REDUCE_MAX, dtypes::Numerical);
DEFINE_CNNL_REDUCE_OP(ReduceMin, CNNL_REDUCE_MIN, dtypes::Numerical);
DEFINE_CNNL_REDUCE_OP(ReduceSum, CNNL_REDUCE_ADD, dtypes::Accumulated);
DEFINE_CNNL_REDUCE_OP(ReduceMean, CNNL_REDUCE_AVG, dtypes::Accumulated);
DEFINE_CNNL_REDUCE_OP(ReduceL1, CNNL_REDUCE_NORM1, dtypes::Floating);
DEFINE_CNNL_REDUCE_OP(ReduceL2, CNNL_REDUCE_NORM2, dtypes::Floating);
DEFINE_CNNL_REDUCE_GRAD_OP(ReduceSum);
DEFINE_CNNL_REDUCE_GRAD_OP(ReduceMean);
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_MATH_REDUCE_OP_H_
