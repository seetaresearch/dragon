// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_VISION_DENSE_BLOCK_OP_H_
#define DRAGON_OPERATORS_VISION_DENSE_BLOCK_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class DenseBlockOp final : public Operator<Context> { 
 public:
    DenseBlockOp(const OperatorDef& op_def, Workspace* ws)
        : num_conv_layers(OperatorBase::GetSingleArg<int>("num_conv_layers", 1)),
          growth_rate(OperatorBase::GetSingleArg<int>("growth_rate", 12)) {}
   
 protected:
     void Init();
     TIndex num_conv_layers, growth_rate;
};



}

#endif  // DRAGON_OPERATORS_VISION_DENSE_BLOCK_OP_H_