/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_MATH_ELEMENTWISE_OPS_H_
#define DRAGON_OPERATORS_MATH_ELEMENTWISE_OPS_H_

#include "dragon/core/operator.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

#define DECLARE_SIMPLE_UNARY_OP(name)               \
  template <class Context>                          \
  class name##Op final : public Operator<Context> { \
   public:                                          \
    SIMPLE_CTOR_DTOR(name##Op);                     \
    USE_OPERATOR_FUNCTIONS;                         \
                                                    \
    void RunOnDevice() override;                    \
                                                    \
    template <typename T>                           \
    void DoRunWithType();                           \
  };

#define DECLARE_SIMPLE_BINARY_OP(name)              \
  template <class Context>                          \
  class name##Op final : public Operator<Context> { \
   public:                                          \
    SIMPLE_CTOR_DTOR(name##Op);                     \
    USE_OPERATOR_FUNCTIONS;                         \
                                                    \
    void RunOnDevice() override;                    \
                                                    \
    template <typename T>                           \
    void DoRunWithType();                           \
  };

inline vec32_t CheckOutputAliases(
    const Tensor& A,
    const Tensor& B,
    Tensor* Y,
    const vec64_t& Y_dims) {
  if ((void*)&A == (void*)Y) {
    CHECK(A.dims() == Y_dims)
        << "\nNon-broadcastable output shape " << A.DimString()
        << " does not match the broadcast shape " << Tensor::DimString(Y_dims);
  } else if ((void*)&B == (void*)Y) {
    CHECK(B.dims() == Y_dims)
        << "\nNon-broadcastable output shape " << B.DimString()
        << " does not match the broadcast shape " << Tensor::DimString(Y_dims);
  }
  vec32_t available_aliases;
  if (Y_dims == A.dims()) available_aliases.push_back(0);
  if (Y_dims == B.dims()) available_aliases.push_back(1);
  return available_aliases;
}

inline void IsBroadcast(
    const Tensor& A,
    const Tensor& B,
    int& rows,
    int& cols,
    int& kind,
    Tensor* Y = nullptr) {
  kind = -2;
  if (A.count() == B.count()) {
    if (Y != nullptr) Y->ReshapeLike(A);
    kind = -1;
  } else if (B.count() < A.count()) {
    if (Y != nullptr) Y->ReshapeLike(A);
    if (utils::math::IsRowwiseBroadcast(A.dims(), B.dims(), &rows, &cols)) {
      kind = 0;
    } else if (utils::math::IsColwiseBroadcast(
                   A.dims(), B.dims(), &rows, &cols)) {
      kind = 1;
    }
  } else {
    if (Y != nullptr) Y->ReshapeLike(B);
    if (utils::math::IsRowwiseBroadcast(A.dims(), B.dims(), &rows, &cols)) {
      kind = 2;
    } else if (utils::math::IsColwiseBroadcast(
                   A.dims(), B.dims(), &rows, &cols)) {
      kind = 3;
    }
  }
}

DECLARE_SIMPLE_UNARY_OP(Abs);
DECLARE_SIMPLE_UNARY_OP(Ceil);
DECLARE_SIMPLE_UNARY_OP(Cos);
DECLARE_SIMPLE_UNARY_OP(Exp);
DECLARE_SIMPLE_UNARY_OP(Floor);
DECLARE_SIMPLE_UNARY_OP(IsInf);
DECLARE_SIMPLE_UNARY_OP(IsNaN);
DECLARE_SIMPLE_UNARY_OP(Log);
DECLARE_SIMPLE_UNARY_OP(Neg);
DECLARE_SIMPLE_UNARY_OP(Invert);
DECLARE_SIMPLE_UNARY_OP(Reciprocal);
DECLARE_SIMPLE_UNARY_OP(Round);
DECLARE_SIMPLE_UNARY_OP(Rsqrt);
DECLARE_SIMPLE_UNARY_OP(Sign);
DECLARE_SIMPLE_UNARY_OP(Sin);
DECLARE_SIMPLE_UNARY_OP(Sqrt);
DECLARE_SIMPLE_UNARY_OP(Square);
DECLARE_SIMPLE_UNARY_OP(AbsGradient);
DECLARE_SIMPLE_UNARY_OP(CosGradient);
DECLARE_SIMPLE_UNARY_OP(ExpGradient);
DECLARE_SIMPLE_UNARY_OP(LogGradient);
DECLARE_SIMPLE_UNARY_OP(NegGradient);
DECLARE_SIMPLE_UNARY_OP(ReciprocalGradient);
DECLARE_SIMPLE_UNARY_OP(RsqrtGradient);
DECLARE_SIMPLE_UNARY_OP(SignGradient);
DECLARE_SIMPLE_UNARY_OP(SinGradient);
DECLARE_SIMPLE_UNARY_OP(SqrtGradient);
DECLARE_SIMPLE_UNARY_OP(SquareGradient);
#undef DECLARE_SIMPLE_UNARY_OP

DECLARE_SIMPLE_BINARY_OP(Add);
DECLARE_SIMPLE_BINARY_OP(Sub);
DECLARE_SIMPLE_BINARY_OP(Mul);
DECLARE_SIMPLE_BINARY_OP(Div);
DECLARE_SIMPLE_BINARY_OP(Pow);
DECLARE_SIMPLE_BINARY_OP(Minimum);
DECLARE_SIMPLE_BINARY_OP(Maximum);
DECLARE_SIMPLE_BINARY_OP(Equal);
DECLARE_SIMPLE_BINARY_OP(NotEqual);
DECLARE_SIMPLE_BINARY_OP(Less);
DECLARE_SIMPLE_BINARY_OP(LessEqual);
DECLARE_SIMPLE_BINARY_OP(Greater);
DECLARE_SIMPLE_BINARY_OP(GreaterEqual);
DECLARE_SIMPLE_BINARY_OP(AddGradient);
DECLARE_SIMPLE_BINARY_OP(SubGradient);
DECLARE_SIMPLE_BINARY_OP(MulGradient);
DECLARE_SIMPLE_BINARY_OP(DivGradient);
DECLARE_SIMPLE_BINARY_OP(PowGradient);
DECLARE_SIMPLE_BINARY_OP(MinimumGradient);
DECLARE_SIMPLE_BINARY_OP(MaximumGradient);
#undef DECLARE_SIMPLE_BINARY_OP

} // namespace dragon

#endif // DRAGON_OPERATORS_MATH_ELEMENTWISE_OPS_H_
