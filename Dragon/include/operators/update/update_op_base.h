// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// -------------------------------------------------------------

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
        slot(OperatorBase::GetSingleArg<string>("slot", "")),
        zero_grad(OperatorBase::GetSingleArg<bool>("zero_grad", true)) {
        CHECK(!slot.empty()) << "\nRequired a non-empty slot";
    }
    USE_OPERATOR_FUNCTIONS;

    float Param(const string& name) const;
    string Slot();

    void RunOnDevice() override;
    template <typename T> void PreprocessRunWithType();
    virtual void ComputeRunWithFloat() = 0;
    virtual void ComputeRunWithFloat16() { LOG(FATAL) << "This Updater does not support FP16."; }
    template <typename T> void UpdateRunWithType();

 protected:
    float lr_mult, decay_mult;
    float l2_decay, clip_thresh, scale_factor;
    string slot;
    bool zero_grad;
};

#define USE_UPDATER_FUNCTIONS(context) \
    using UpdateOpBase<context>::Param; \
    using UpdateOpBase<context>::Slot

}    // namespace dragon 

#endif    // DRAGON_OPERATORS_UPDATE_UPDATE_OP_BASE_H_