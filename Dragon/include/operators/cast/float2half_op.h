// ------------------------------------------------------------
// Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_OPERATORS_CAST_FLOAT2HALF_OP_H_
#define DRAGON_OPERATORS_CAST_FLOAT2HALF_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class FloatToHalfOp final : public Operator<Context> {
 public:
     USE_SIMPLE_CTOR_DTOR(FloatToHalfOp);
     USE_OPERATOR_FUNCTIONS(Context);
     void RunOnDevice() override;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_CAST_FLOAT2HALF_OP_H_