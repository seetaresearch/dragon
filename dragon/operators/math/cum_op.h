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

#ifndef DRAGON_OPERATORS_MATH_CUM_OP_H_
#define DRAGON_OPERATORS_MATH_CUM_OP_H_

#include "dragon/core/operator.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {

template <class Context, class Functor>
class CumOp : public Operator<Context> {
 public:
  CumOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        exclusive_(OP_SINGLE_ARG(int64_t, "exclusive", 0)),
        reverse_(OP_SINGLE_ARG(int64_t, "reverse", 0)) {}
  USE_OPERATOR_FUNCTIONS;

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t exclusive_, reverse_;
  Functor functor_;
};

namespace {

#define DEFINE_FUNCTOR(name, Functor)                         \
  struct name {                                               \
    template <typename T, class Context>                      \
    inline void Compute(                                      \
        const int N,                                          \
        const int S,                                          \
        const int C,                                          \
        const bool exclusive,                                 \
        const bool reverse,                                   \
        const T* x,                                           \
        T* y,                                                 \
        Context* ctx) const {                                 \
      return Functor(N, S, C, exclusive, reverse, x, y, ctx); \
    }                                                         \
  };

DEFINE_FUNCTOR(CumSumFunctor, kernels::CumSum);
DEFINE_FUNCTOR(CumMaxFunctor, kernels::CumMax);
DEFINE_FUNCTOR(CumMinFunctor, kernels::CumMin);
#undef DEFINE_FUNCTOR

} // namespace

#define DEFINE_CUM_OP(name, BaseOp, Functor, ReduceTypes)  \
  template <class Context>                                 \
  class name##Op final : public BaseOp<Context, Functor> { \
   public:                                                 \
    name##Op(const OperatorDef& def, Workspace* ws)        \
        : BaseOp<Context, Functor>(def, ws) {}             \
    USE_OPERATOR_FUNCTIONS;                                \
                                                           \
    void RunOnDevice() override {                          \
      DispatchHelper<ReduceTypes>::Call(this, Input(0));   \
    }                                                      \
  };

DEFINE_CUM_OP(CumSum, CumOp, CumSumFunctor, dtypes::Numerical);
DEFINE_CUM_OP(CumMax, CumOp, CumMaxFunctor, dtypes::Numerical);
DEFINE_CUM_OP(CumMin, CumOp, CumMinFunctor, dtypes::Numerical);

#define DECLARE_CUM_GRAD_OP(name)                             \
  template <class Context>                                    \
  class name##Op final : public Operator<Context> {           \
   public:                                                    \
    name##Op(const OperatorDef& def, Workspace* ws)           \
        : Operator<Context>(def, ws),                         \
          exclusive_(OP_SINGLE_ARG(int64_t, "exclusive", 0)), \
          reverse_(OP_SINGLE_ARG(int64_t, "reverse", 0)) {}   \
    USE_OPERATOR_FUNCTIONS;                                   \
                                                              \
    void RunOnDevice() override {                             \
      DispatchHelper<dtypes::Floating>::Call(this, Input(0)); \
    }                                                         \
                                                              \
    template <typename T>                                     \
    void DoRunWithType();                                     \
                                                              \
   protected:                                                 \
    int64_t exclusive_, reverse_;                             \
  };

DECLARE_CUM_GRAD_OP(CumSumGradient);
#undef DECLARE_CUM_GRAD_OP

#ifdef USE_MLU
#define DEFINE_CNNL_CUM_OP(name, ReduceTypes)                 \
  template <class Context>                                    \
  class CNNL##name##Op final : public Operator<Context> {     \
   public:                                                    \
    CNNL##name##Op(const OperatorDef& def, Workspace* ws)     \
        : Operator<Context>(def, ws),                         \
          exclusive_(OP_SINGLE_ARG(int64_t, "exclusive", 0)), \
          reverse_(OP_SINGLE_ARG(int64_t, "reverse", 0)) {    \
      CNNLCreateTensorDesc(&input_desc_);                     \
    }                                                         \
    USE_OPERATOR_FUNCTIONS;                                   \
    ~CNNL##name##Op() {                                       \
      CNNLDestroyTensorDesc(input_desc_);                     \
    }                                                         \
    void RunOnDevice() override {                             \
      DispatchHelper<ReduceTypes>::Call(this, Input(0));      \
    }                                                         \
    template <typename T>                                     \
    void DoRunWithType();                                     \
                                                              \
   protected:                                                 \
    int64_t exclusive_, reverse_;                             \
    cnnlTensorDescriptor_t input_desc_;                       \
  };

DEFINE_CNNL_CUM_OP(CumSum, dtypes::Numerical);
DEFINE_CNNL_CUM_OP(CumSumGradient, dtypes::Floating);
#undef DEFINE_CNNL_CUM_OP
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_MATH_CUM_OP_H_
