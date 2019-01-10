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
#include "utils/math_utils.h"

namespace dragon {

#define DECLARE_FUNDAMENTAL_OP(type) \
    template <class Context> \
    class type##Op final : public Operator<Context> { \
     public: \
         USE_SIMPLE_CTOR_DTOR(type##Op); \
         USE_OPERATOR_FUNCTIONS; \
         void RunOnDevice() override; \
         template <typename T> void EltwiseRunWithType(); \
         template <typename T> void BroadcastRunWithType(int type); \
     protected: \
        int rows, cols; \
    };

DECLARE_FUNDAMENTAL_OP(Add);
DECLARE_FUNDAMENTAL_OP(Sub);
DECLARE_FUNDAMENTAL_OP(Mul);
DECLARE_FUNDAMENTAL_OP(Div);

DECLARE_FUNDAMENTAL_OP(RAdd);
DECLARE_FUNDAMENTAL_OP(RSub);
DECLARE_FUNDAMENTAL_OP(RMul);
DECLARE_FUNDAMENTAL_OP(RDiv);

DECLARE_FUNDAMENTAL_OP(AddGradient);
DECLARE_FUNDAMENTAL_OP(SubGradient);
DECLARE_FUNDAMENTAL_OP(MulGradient);
DECLARE_FUNDAMENTAL_OP(DivGradient);

DECLARE_FUNDAMENTAL_OP(RAddGradient);
DECLARE_FUNDAMENTAL_OP(RSubGradient);
DECLARE_FUNDAMENTAL_OP(RMulGradient);
DECLARE_FUNDAMENTAL_OP(RDivGradient);

#define DECLARE_FUNDAMENTAL_OP_X1X2 \
    ws()->CreateTensor(mount_name( \
        "fundamental/X1"))->ReshapeLike(Input(0)); \
    ws()->CreateTensor(mount_name( \
        "fundamental/X2"))->ReshapeLike(Input(1));

#define DEFINE_FUNDAMENTAL_OP_X1X2 \
    Tensor* X1 = ws()->GetTensor(mount_name("fundamental/X1")); \
    Tensor* X2 = ws()->GetTensor(mount_name("fundamental/X2"));

#define DEFINE_FUNDAMENTAL_TYPED_CALLER(dtype) \
    DEFINE_FUNDAMENTAL_OP_X1X2; \
    if (X2->count() < X1->count() && \
        utils::IsRowwiseBroadcast( \
            X1->dims(), X2->dims(), &rows, &cols)) { \
        BroadcastRunWithType<dtype>(0); \
    } else if (X2->count() < X1->count() && \
       utils::IsColwiseBroadcast( \
           X1->dims(), X2->dims(), &rows, &cols)) { \
        BroadcastRunWithType<dtype>(1); \
    } else if (X1->count() == X2->count()) { \
        EltwiseRunWithType<dtype>(); \
    } else { \
        LOG(FATAL) << "Could not broadcast with shapes: " \
                   << X1->DimString() << " and " \
                   << X2->DimString(); \
    }

#define DEFINE_FUNDAMENTAL_TYPED_RCALLER(dtype) \
    DEFINE_FUNDAMENTAL_OP_X1X2; \
    if (X2->count() > X1->count() && \
        utils::IsRowwiseBroadcast( \
            X1->dims(), X2->dims(), &rows, &cols)) { \
        BroadcastRunWithType<dtype>(2); \
    } else if (X2->count() > X1->count() && \
       utils::IsColwiseBroadcast( \
           X1->dims(), X2->dims(), &rows, &cols)) { \
        BroadcastRunWithType<dtype>(3); \
    } else if (X1->count() == X2->count()) { \
        EltwiseRunWithType<dtype>(); \
    } else { \
        LOG(FATAL) << "Could not broadcast with shapes: " \
                   << X1->DimString() << " and " \
                   << X2->DimString(); \
    }

#undef DECLARE_FUNDAMENTAL_OP

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARITHMETIC_FUNDAMENTAL_OP_H_