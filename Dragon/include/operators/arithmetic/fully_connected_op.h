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

#ifndef DRAGON_OPERATORS_ARITHMETIC_FULLY_CONNECTED_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_FULLY_CONNECTED_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class FullyConnectedOp final : public Operator<Context> {
 public:
    FullyConnectedOp(const OperatorDef& def, Workspace *ws)
        : Operator<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", 1)),
          N_(OpArg<int64_t>("num_output", 0)),
          transW_(OpArg<bool>("transW", true)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice();
    template <typename T> void TransRunImpl();
    template <typename T> void NoTransRunImpl();

 protected:
    int64_t axis_, transW_, M_, K_, N_;
};

template <class Context>
class FullyConnectedGradientOp
    final : public Operator<Context> {
 public:
    FullyConnectedGradientOp(
        const OperatorDef&      def,
        Workspace*              ws)
        : Operator<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", 1)),
          N_(OpArg<int64_t>("num_output", 0)),
          transW_(OpArg<bool>("transW", true)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    int64_t axis_, transW_, M_, K_, N_;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARITHMETIC_FULLY_CONNECTED_OP_H_