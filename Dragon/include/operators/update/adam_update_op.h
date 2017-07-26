// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_UPDATE_ADAM_UPDATE_OP_H_
#define DRAGON_OPERATORS_UPDATE_ADAM_UPDATE_OP_H_

#include "operators/update/update_op_base.h"

namespace dragon {

template <class Context>
class AdamUpdateOp final : public UpdateOpBase<Context> {
 public:
    AdamUpdateOp(const OperatorDef& op_def, Workspace* ws) 
        : UpdateOpBase<Context>(op_def, ws), 
          t(0), 
          eps(param("eps")),
          beta1(param("beta1")), 
          beta2(param("beta2")) {}

    void ComputeRunWithFloat() override;

 protected:
    unique_ptr<Tensor> m, v, tmp;
    float lr, beta1, beta2, eps, coeff;
    int t;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_UPDATE_ADAM_UPDATE_OP_H_