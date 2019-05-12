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

#ifndef DRAGON_OPERATORS_ARRAY_REDUCE_OP_H_
#define DRAGON_OPERATORS_ARRAY_REDUCE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ReduceOp final : public Operator<Context> {
 public:
    ReduceOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axes_(OpArgs<int64_t>("axes")),
          keep_dims_(OpArg<int64_t>("keep_dims", 0)),
          operation_(OpArg<string>("operation", "SUM")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    string operation_;
    int64_t keep_dims_;
    vec64_t dims_, axes_;
    vec32_t dims32_, axes32_;
};

template <class Context>
class ReduceGradientOp final : public Operator<Context> {
 public:
    ReduceGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axes_(OpArgs<int64_t>("axes")),
          operation_(OpArg<string>("operation", "SUM")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    string operation_;
    vec32_t axes32_;
    vec64_t axes_, y_dims_, y_strides_;
    Tensor X_dims_, Y_dims_, Y_strides_;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARRAY_REDUCE_OP_H_