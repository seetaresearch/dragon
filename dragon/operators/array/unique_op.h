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

#ifndef DRAGON_OPERATORS_ARRAY_UNIQUE_OP_H_
#define DRAGON_OPERATORS_ARRAY_UNIQUE_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class UniqueOp final : public Operator<Context> {
 public:
  UniqueOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        return_inverse_(OP_SINGLE_ARG(int64_t, "return_inverse", 0)),
        return_counts_(OP_SINGLE_ARG(int64_t, "return_counts", 0)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t return_inverse_, return_counts_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_UNIQUE_OP_H_
