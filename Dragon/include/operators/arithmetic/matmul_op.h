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

#ifndef DRAGON_OPERATORS_ARITHMETIC_MATMUL_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_MATMUL_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class MatmulOp final : public Operator<Context> {
 public:
    MatmulOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          transA(OperatorBase::Arg<bool>("transA", false)),
          transB(OperatorBase::Arg<bool>("transB", false)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    int64_t M1, N1, M2, N2;
    int64_t transA, transB, M, K1, K2, N;
    int64_t batch_size, A_stride, B_stride, C_stride;
};

template <class Context>
class MatmulGradientOp final : public Operator<Context> {
 public:
    MatmulGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          transA(OperatorBase::Arg<bool>("transA", false)),
          transB(OperatorBase::Arg<bool>("transB", false)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    int64_t M1, N1, M2, N2;
    int64_t transA, transB, M, K1, K2, N;
    int64_t batch_size, A_stride, B_stride, C_stride;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARITHMETIC_MATMUL_OP_H_