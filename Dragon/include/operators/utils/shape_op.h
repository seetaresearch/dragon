// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_UTILS_SHAPE_OP_H_
#define DRAGON_OPERATORS_UTILS_SHAPE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ShapeOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(ShapeOp);
    void RunOnDevice() override;
};

}    // namespace dragon

#endif    //DRAGON_OPERATORS_UTILS_SHAPE_OP_H_