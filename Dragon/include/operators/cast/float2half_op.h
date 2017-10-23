// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_CAST_FLOAT2HALF_OP_H_
#define DRAGON_OPERATORS_CAST_FLOAT2HALF_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class FloatToHalfOp final : public Operator<Context> {
 public:
     USE_SIMPLE_CTOR_DTOR(FloatToHalfOp);
     void RunOnDevice() override;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_CAST_FLOAT2HALF_OP_H_