// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_UPDATE_UPDATE_OP_BASE_H_
#define DRAGON_OPERATORS_UPDATE_UPDATE_OP_BASE_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class UpdateOpBase : public Operator<Context> {
 public:
    UpdateOpBase(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          lr_mult(OperatorBase::GetSingleArg<float>("lr_mult", 1.0)),
          decay_mult(OperatorBase::GetSingleArg<float>("decay_mult", 1.0)),
          domain(OperatorBase::GetSingleArg<string>("domain", "_")) {}

    float param(const string& name) const;

    void RunOnDevice() override;
    template <typename T> void PreprocessRunWithType();
    virtual void ComputeRunWithFloat() = 0;
    template <typename T> void UpdateRunWithType();

 protected:
    float lr_mult, decay_mult;
    float l2_decay, clip_thresh, scale_factor;
    string domain;
};

}    // namespace dragon 

#endif    // DRAGON_OPERATORS_UPDATE_UPDATE_OP_BASE_H_