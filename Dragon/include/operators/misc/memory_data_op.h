// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_MISC_MEMORY_DATA_OP_H_
#define DRAGON_OPERATORS_MISC_MEMORY_DATA_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class MemoryDataOp final : public Operator<Context> {
 public:
     MemoryDataOp(const OperatorDef& op_def, Workspace* ws)
         : Operator<Context>(op_def, ws) {
         int DATA_TYPE = OperatorBase::GetSingleArg<int>("dtype", 1);
         data_type = TensorProto_DataType(DATA_TYPE);
     }

    void RunOnDevice() override;
    template <typename Tx, typename Ty> void RunWithType();

 protected:
     TensorProto_DataType data_type;

};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_MISC_MEMORY_DATA_OP_H_