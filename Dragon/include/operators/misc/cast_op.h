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

#ifndef DRAGON_OPERATORS_MISC_CAST_OP_H_
#define DRAGON_OPERATORS_MISC_CAST_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class CastOp final : public Operator<Context> {
 public:
    CastOp(const OperatorDef& def, Workspace* ws)
       : Operator<Context>(def, ws),
         inplace_(OpArg<int64_t>("inplace", 0)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;

 protected:
    int64_t inplace_;
};

template <class Context>
class CastGradientOp final : public Operator<Context> {
 public:
    SIMPLE_CTOR_DTOR(CastGradientOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_MISC_CAST_OP_H_