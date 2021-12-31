/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_MATH_ELEMENTWISE_OP_H_
#define DRAGON_OPERATORS_MATH_ELEMENTWISE_OP_H_

#include "dragon/core/operator.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

#define DECLARE_ELEMENTWISE_OP(name)                \
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

template <class Context>
class AxpbyOp final : public Operator<Context> {
 public:
  AxpbyOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        alpha_(OP_SINGLE_ARG(float, "alpha", 1.f)),
        beta_(OP_SINGLE_ARG(float, "beta", 1.f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float alpha_, beta_;
  Tensor *X_, *Y_;
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

// Unary ElementwiseOp.
DECLARE_ELEMENTWISE_OP(Abs);
DECLARE_ELEMENTWISE_OP(Ceil);
DECLARE_ELEMENTWISE_OP(Cos);
DECLARE_ELEMENTWISE_OP(Exp);
DECLARE_ELEMENTWISE_OP(Floor);
DECLARE_ELEMENTWISE_OP(IsInf);
DECLARE_ELEMENTWISE_OP(IsNaN);
DECLARE_ELEMENTWISE_OP(IsFinite);
DECLARE_ELEMENTWISE_OP(Log);
DECLARE_ELEMENTWISE_OP(Neg);
DECLARE_ELEMENTWISE_OP(Reciprocal);
DECLARE_ELEMENTWISE_OP(Round);
DECLARE_ELEMENTWISE_OP(Rsqrt);
DECLARE_ELEMENTWISE_OP(Sign);
DECLARE_ELEMENTWISE_OP(Sin);
DECLARE_ELEMENTWISE_OP(Sqrt);
DECLARE_ELEMENTWISE_OP(Square);
DECLARE_ELEMENTWISE_OP(BitwiseNot);
DECLARE_ELEMENTWISE_OP(Not);
DECLARE_ELEMENTWISE_OP(AbsGradient);
DECLARE_ELEMENTWISE_OP(CosGradient);
DECLARE_ELEMENTWISE_OP(ExpGradient);
DECLARE_ELEMENTWISE_OP(LogGradient);
DECLARE_ELEMENTWISE_OP(NegGradient);
DECLARE_ELEMENTWISE_OP(ReciprocalGradient);
DECLARE_ELEMENTWISE_OP(RsqrtGradient);
DECLARE_ELEMENTWISE_OP(SignGradient);
DECLARE_ELEMENTWISE_OP(SinGradient);
DECLARE_ELEMENTWISE_OP(SqrtGradient);
DECLARE_ELEMENTWISE_OP(SquareGradient);
// Binary ElementwiseOp.
DECLARE_ELEMENTWISE_OP(Add);
DECLARE_ELEMENTWISE_OP(Sub);
DECLARE_ELEMENTWISE_OP(Mul);
DECLARE_ELEMENTWISE_OP(Div);
DECLARE_ELEMENTWISE_OP(Pow);
DECLARE_ELEMENTWISE_OP(Atan2);
DECLARE_ELEMENTWISE_OP(Minimum);
DECLARE_ELEMENTWISE_OP(Maximum);
DECLARE_ELEMENTWISE_OP(BitwiseAnd);
DECLARE_ELEMENTWISE_OP(BitwiseOr);
DECLARE_ELEMENTWISE_OP(BitwiseXor);
DECLARE_ELEMENTWISE_OP(And);
DECLARE_ELEMENTWISE_OP(Or);
DECLARE_ELEMENTWISE_OP(Xor);
DECLARE_ELEMENTWISE_OP(Equal);
DECLARE_ELEMENTWISE_OP(NotEqual);
DECLARE_ELEMENTWISE_OP(Less);
DECLARE_ELEMENTWISE_OP(LessEqual);
DECLARE_ELEMENTWISE_OP(Greater);
DECLARE_ELEMENTWISE_OP(GreaterEqual);
DECLARE_ELEMENTWISE_OP(AddGradient);
DECLARE_ELEMENTWISE_OP(SubGradient);
DECLARE_ELEMENTWISE_OP(MulGradient);
DECLARE_ELEMENTWISE_OP(DivGradient);
DECLARE_ELEMENTWISE_OP(PowGradient);
DECLARE_ELEMENTWISE_OP(MinimumGradient);
DECLARE_ELEMENTWISE_OP(MaximumGradient);
// Trinary ElementwiseOp.
DECLARE_ELEMENTWISE_OP(Where);
DECLARE_ELEMENTWISE_OP(WhereGradient);
#undef DECLARE_ELEMENTWISE_OP

} // namespace dragon

#endif // DRAGON_OPERATORS_MATH_ELEMENTWISE_OP_H_
