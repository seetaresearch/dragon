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
          transA_(OpArg<bool>("transA", false)),
          transB_(OpArg<bool>("transB", false)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    int64_t batch_size_;
    int64_t transA_, transB_;
    int64_t M_, K1_, K2_, N_;
    int64_t M1_, N1_, M2_, N2_;
    int64_t A_stride_, B_stride_, Y_stride_;
};

template <class Context>
class MatmulGradientOp final : public Operator<Context> {
 public:
    MatmulGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          transA_(OpArg<bool>("transA", false)),
          transB_(OpArg<bool>("transB", false)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    int64_t batch_size_;
    int64_t transA_, transB_;
    int64_t M_, K1_, K2_, N_;
    int64_t M1_, N1_, M2_, N2_;
    int64_t A_stride_, B_stride_, Y_stride_;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARITHMETIC_MATMUL_OP_H_