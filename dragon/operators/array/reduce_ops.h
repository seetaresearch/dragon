/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_ARRAY_REDUCE_OPS_H_
#define DRAGON_OPERATORS_ARRAY_REDUCE_OPS_H_

#include "dragon/core/operator.h"

namespace dragon {

#define DECLARE_REDUCE_OP(name)                         \
  template <class Context>                              \
  class name##Op final : public Operator<Context> {     \
   public:                                              \
    name##Op(const OperatorDef& def, Workspace* ws)     \
        : Operator<Context>(def, ws),                   \
          axes_(OpArgs<int64_t>("axes")),               \
          keep_dims_(OpArg<int64_t>("keep_dims", 0)) {} \
    USE_OPERATOR_FUNCTIONS;                             \
                                                        \
    void RunOnDevice() override;                        \
                                                        \
    template <typename T>                               \
    void DoRunWithType();                               \
                                                        \
   protected:                                           \
    int64_t keep_dims_;                                 \
    vec64_t axes_;                                      \
  };

#define DECLARE_REDUCE_GRAD_OP(name)                        \
  template <class Context>                                  \
  class name##GradientOp final : public Operator<Context> { \
   public:                                                  \
    SIMPLE_CTOR_DTOR(name##GradientOp);                     \
    USE_OPERATOR_FUNCTIONS;                                 \
                                                            \
    void RunOnDevice() override;                            \
                                                            \
    template <typename T>                                   \
    void DoRunWithType();                                   \
  };

DECLARE_REDUCE_OP(ReduceMax);
DECLARE_REDUCE_OP(ReduceMean);
DECLARE_REDUCE_OP(ReduceMin);
DECLARE_REDUCE_OP(ReduceSum);

DECLARE_REDUCE_GRAD_OP(ReduceMean);
DECLARE_REDUCE_GRAD_OP(ReduceSum);

#undef DECLARE_REDUCE_OP
#undef DECLARE_REDUCE_GRAD_OP

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_REDUCE_OPS_H_
