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

#ifndef DRAGON_OPERATORS_ARRAY_INDEX_SELECT_OP_H_
#define DRAGON_OPERATORS_ARRAY_INDEX_SELECT_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class IndexSelectOp final : public Operator<Context> {
 public:
    IndexSelectOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", 0)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    int64_t outer_dim_, inner_dim_;
    int64_t axis_, axis_dim_, nindices_;
};

template <class Context>
class IndexSelectGradientOp final : public Operator<Context> {
 public:
    IndexSelectGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", 0)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    int64_t outer_dim_, inner_dim_;
    int64_t axis_, axis_dim_, nindices_;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARRAY_INDEX_SELECT_OP_H_