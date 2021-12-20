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

namespace dragon {

#define DECLARE_CUM_OP(name)                                  \
  template <class Context>                                    \
  class name##Op final : public Operator<Context> {           \
   public:                                                    \
    name##Op(const OperatorDef& def, Workspace* ws)           \
        : Operator<Context>(def, ws),                         \
          exclusive_(OP_SINGLE_ARG(int64_t, "exclusive", 0)), \
          reverse_(OP_SINGLE_ARG(int64_t, "reverse", 0)) {}   \
    USE_OPERATOR_FUNCTIONS;                                   \
                                                              \
    void RunOnDevice() override;                              \
                                                              \
    template <typename T>                                     \
    void DoRunWithType();                                     \
                                                              \
   protected:                                                 \
    int64_t exclusive_, reverse_;                             \
  };

DECLARE_CUM_OP(CumSum);
DECLARE_CUM_OP(CumSumGradient);

#undef DECLARE_CUM_OP

} // namespace dragon

#endif // DRAGON_OPERATORS_MATH_CUM_OP_H_
