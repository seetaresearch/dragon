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

#define DECLARE_FUNDAMENTAL_OP(name) \
    template <class Context> \
    class name##Op final : public Operator<Context> { \
     public: \
         SIMPLE_CTOR_DTOR(name##Op); \
         USE_OPERATOR_FUNCTIONS; \
         void RunOnDevice() override; \
         template <typename T> void EltwiseRunImpl(); \
         template <typename T> void BroadcastRunImpl(int type); \
     protected: \
        int rows_, cols_; \
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

#define DECLARE_INPUT_DESC \
    ws()->CreateTensor(unique_name("A"))->ReshapeLike(X(0)); \
    ws()->CreateTensor(unique_name("B"))->ReshapeLike(X(1));

#define DEFINE_INPUT_DESC \
    auto* A = ws()->GetTensor(unique_name("A")); \
    auto* B = ws()->GetTensor(unique_name("B"));

#define DEFINE_FUNDAMENTAL_TYPED_IMPL(dtype) \
    DEFINE_INPUT_DESC; \
    if (B->count() < A->count() && \
        utils::IsRowwiseBroadcast( \
            A->dims(), B->dims(), &rows_, &cols_)) { \
        BroadcastRunImpl<dtype>(0); \
    } else if (B->count() < A->count() && \
       utils::IsColwiseBroadcast( \
           A->dims(), B->dims(), &rows_, &cols_)) { \
        BroadcastRunImpl<dtype>(1); \
    } else if (A->count() == B->count()) { \
        EltwiseRunImpl<dtype>(); \
    } else { \
        LOG(FATAL) << "Could not broadcast with shapes: " \
                   << A->DimString() << " and " \
                   << B->DimString(); \
    }

#define DEFINE_RFUNDAMENTAL_TYPED_IMPL(dtype) \
    DEFINE_INPUT_DESC; \
    if (B->count() > A->count() && \
        utils::IsRowwiseBroadcast( \
            A->dims(), B->dims(), &rows_, &cols_)) { \
        BroadcastRunImpl<dtype>(2); \
    } else if (B->count() > A->count() && \
       utils::IsColwiseBroadcast( \
           A->dims(), B->dims(), &rows_, &cols_)) { \
        BroadcastRunImpl<dtype>(3); \
    } else if (A->count() == B->count()) { \
        EltwiseRunImpl<dtype>(); \
    } else { \
        LOG(FATAL) << "Could not broadcast with shapes: " \
                   << A->DimString() << " and " \
                   << B->DimString(); \
    }

#undef DECLARE_FUNDAMENTAL_OP

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARITHMETIC_FUNDAMENTAL_OP_H_