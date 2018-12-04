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

#ifndef DRAGON_OPERATORS_ARITHMETIC_FUNDAMENTAL_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_FUNDAMENTAL_OP_H_

#include "core/operator.h"

namespace dragon {

/*********************************************
*                                            *
*                    Add                     *
*                                            *
**********************************************/

template <class Context>
class AddOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(AddOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);
};

template <class Context>
class AddGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(AddGradientOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);

 protected:
    Tensor *X1, *X2;
};

/*********************************************
*                                            *
*                    RAdd                    *
*                                            *
**********************************************/

template <class Context>
class RAddOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(RAddOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);
};

template <class Context>
class RAddGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(RAddGradientOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);
};

/*********************************************
*                                            *
*                     Sub                    *
*                                            *
**********************************************/

template <class Context>
class SubOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(SubOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);
};

template <class Context>
class SubGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(SubGradientOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);
};

/*********************************************
*                                            *
*                    RSub                    *
*                                            *
**********************************************/

template <class Context>
class RSubOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(RSubOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);
};

template <class Context>
class RSubGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(RSubGradientOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);
};

/*********************************************
*                                            *
*                     Mul                    *
*                                            *
**********************************************/

template <class Context>
class MulOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(MulOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);
};

template <class Context>
class MulGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(MulGradientOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);
};

/*********************************************
*                                            *
*                     RMul                   *
*                                            *
**********************************************/

template <class Context>
class RMulOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(RMulOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);
};

template <class Context>
class RMulGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(RMulGradientOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);
};

/*********************************************
*                                            *
*                    Div                     *
*                                            *
**********************************************/

template <class Context>
class DivOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(DivOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);
};

template <class Context>
class DivGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(DivGradientOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);
};

/*********************************************
*                                            *
*                    RDiv                    *
*                                            *
**********************************************/

template <class Context>
class RDivOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(RDivOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);
};

template <class Context>
class RDivGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(RDivGradientOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType(int type);
};

#define DeclareX1X2 \
    ws()->CreateTensor( \
        "/mnt/" + anchor() + "/fundamental/X1") \
        ->ReshapeLike(Input(0)); \
    ws()->CreateTensor( \
        "/mnt/" + anchor() + "/fundamental/X2") \
        ->ReshapeLike(Input(1))

#define DefineX1X2 \
    Tensor* X1 = ws()->GetTensor( \
        "/mnt/" + anchor() + "/fundamental/X1"); \
    Tensor* X2 = ws()->GetTensor( \
        "/mnt/" + anchor() + "/fundamental/X2")

#define RunByX1X2(dtype) \
    DefineX1X2; \
    if (X2->ndim() == 0) { \
        BroadcastRunWithType<dtype>(0); \
    } else if (X2->ndim() == 1 && X2->dim(0) == 1) { \
        BroadcastRunWithType<dtype>(0); \
    } else if (X1->dim(-1) == X2->dim(-1) && \
        X2->count(0, X2->axis(-1)) == 1) { \
        BroadcastRunWithType<dtype>(1); \
    } else if (X1->dim(0) == X2->dim(0) && \
               X2->count(1) == 1) { \
        BroadcastRunWithType<dtype>(2); \
    } else if (X1->dims() == X2->dims()) { \
        EltwiseRunWithType<dtype>(); \
    } else { \
        LOG(FATAL) << "Could not broadcast with shapes " \
                   << X1->DimString() << "  " \
                   << X2->DimString(); \
    }

#define RRunByX1X2(dtype) \
    DefineX1X2; \
    if (X1->ndim() == 0) { \
        BroadcastRunWithType<dtype>(0); \
    } else if (X1->ndim() == 1 && X1->dim(0) == 1) { \
        BroadcastRunWithType<dtype>(0); \
    } else if (X1->dim(-1) == X2->dim(-1) && \
        X1->count(0, X1->axis(-1)) == 1) { \
        BroadcastRunWithType<dtype>(1); \
    } else if (X1->dim(0) == X2->dim(0) && \
        X1->count(1) == 1) { \
        BroadcastRunWithType<dtype>(2); \
    } else if (X1->dims() == X2->dims()) { \
        EltwiseRunWithType<dtype>(); \
    } else { \
        LOG(FATAL) << "Could not broadcast with shapes " \
                   << X1->DimString() << "  " \
                   << X2->DimString(); \
    }

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARITHMETIC_FUNDAMENTAL_OP_H_