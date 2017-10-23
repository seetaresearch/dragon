// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_CONTROL_FLOW_COPY_OP_H_
#define DRAGON_OPERATORS_CONTROL_FLOW_COPY_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class CopyOp final : public Operator<Context> {
 public:
     USE_SIMPLE_CTOR_DTOR(CopyOp);
     void RunOnDevice() override;
     template <typename T> void RunWithType();
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_CONTROL_FLOW_COPY_OP_H_