// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_UPDATE_SGD_UPDATE_OP_H_
#define DRAGON_OPERATORS_UPDATE_SGD_UPDATE_OP_H_

#include "operators/update/update_op_base.h"

namespace dragon {

template <class Context>
class SGDUpdateOp final : public UpdateOpBase<Context> {
 public:
    SGDUpdateOp(const OperatorDef& op_def, Workspace* ws) 
        : UpdateOpBase<Context>(op_def, ws),
          momentum(param("momentum")) {}

    void ComputeRunWithFloat() override;

 protected:    
    float lr, momentum;
    unique_ptr<Tensor> history;

};

}    // namespace dragon 

#endif    // DRAGON_OPERATORS_UPDATE_SGD_UPDATE_OP_H_