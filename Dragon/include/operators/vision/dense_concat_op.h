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

#ifndef DRAGON_OPERATORS_VISION_DENSE_CONCAT_OP_H_
#define DRAGON_OPERATORS_VISION_DENSE_CONCAT_OP_H_

#include "operators/ndarray/concat_op.h"

namespace dragon {

template <class Context>
class DenseConcatOp final : public ConcatOp<Context> {
 public:
    DenseConcatOp(const OperatorDef& def, Workspace* ws)
        : ConcatOp<Context>(def, ws) {}
    USE_OPERATOR_FUNCTIONS;
};

template <class Context>
class DenseConcatGradientOp final : public ConcatGradientOp<Context> {
 public:
    DenseConcatGradientOp(const OperatorDef& def, Workspace* ws)
        : ConcatGradientOp<Context>(def, ws),
          growth_rate(OperatorBase::Arg<int64_t>("growth_rate", 0)) {}
    USE_OPERATOR_FUNCTIONS;

    void ElimateCorruption() override;
    template <typename T> void RestoreX1();

 protected:
    int64_t growth_rate;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_VISION_DENSE_CONCAT_OP_H_