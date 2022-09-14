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

using ReduceVarTypes = dtypes::TypesBase<float, double>;

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

#define DEFINE_REDUCE_OP(name, Reducer, ReduceTypes)         \
  template <class Context>                                   \
  class name##Op final : public ReduceOp<Context, Reducer> { \
   public:                                                   \
    name##Op(const OperatorDef& def, Workspace* ws)          \
        : ReduceOp<Context, Reducer>(def, ws) {}             \
    USE_OPERATOR_FUNCTIONS;                                  \
                                                             \
    void RunOnDevice() override {                            \
      DispatchHelper<ReduceTypes>::Call(this, Input(0));     \
    }                                                        \
  };

DEFINE_REDUCE_OP(ReduceMax, MaxReducer, dtypes::Numerical);
DEFINE_REDUCE_OP(ReduceMin, MinReducer, dtypes::Numerical);
DEFINE_REDUCE_OP(ReduceSum, SumReducer, dtypes::Accumulated);
DEFINE_REDUCE_OP(ReduceMean, MeanReducer, dtypes::Accumulated);
DEFINE_REDUCE_OP(ReduceVar, VarReducer, ReduceVarTypes);
DEFINE_REDUCE_OP(ReduceL1, L1Reducer, dtypes::Floating);
DEFINE_REDUCE_OP(ReduceL2, L2Reducer, dtypes::Floating);
#undef DEFINE_REDUCE_OP

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
#undef DEFINE_REDUCE_GRAD_OP

} // namespace dragon

#endif // DRAGON_OPERATORS_MATH_REDUCE_OP_H_
