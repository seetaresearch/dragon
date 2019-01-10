/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_ARITHMETIC_MOMENTS_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_MOMENTS_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class MomentsOp final : public Operator<Context> {
 public:
    MomentsOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axes(OperatorBase::Args<int64_t>("axes")),
          keep_dims(OperatorBase::Arg<int64_t>("keep_dims", 0)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename Tx, typename Ty> void RunWithType();

 protected:
    int64_t keep_dims;
    vector<int64_t> dims, axes;
    vector<int> dims32, axes32;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARITHMETIC_MOMENTS_OP_H_