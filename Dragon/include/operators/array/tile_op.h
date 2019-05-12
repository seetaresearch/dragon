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

#ifndef DRAGON_OPERATORS_ARRAY_TILE_OP_H_
#define DRAGON_OPERATORS_ARRAY_TILE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class TileOp final : public Operator<Context> {
 public:
    TileOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws) {
        GET_ARGS_WITH_DESC(int64_t, multiples);
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template<typename T> void RunImpl();

 protected:
    Tensor X_dims_, X_strides_, Y_dims_;
    DECLARE_ARGS_WITH_DESC(int64_t, multiples);
};

template <class Context>
class TileGradientOp final : public Operator<Context> {
 public:
    TileGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws) {
        GET_ARGS_WITH_DESC(int64_t, multiples);
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template<typename T> void RunImpl();

 protected:
    Tensor* dst_, * src_, nav_;
    int64_t axis_, multiple_, rows_, cols_;
    DECLARE_ARGS_WITH_DESC(int64_t, multiples);
};

DEFINE_ARGS_WITH_DESC(int64_t, TileOp, multiples);
DEFINE_ARGS_WITH_DESC(int64_t, TileGradientOp, multiples);

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARRAY_TILE_OP_H_