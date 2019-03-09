/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_UPDATE_UPDATE_OP_BASE_H_
#define DRAGON_OPERATORS_UPDATE_UPDATE_OP_BASE_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class UpdateOpBase : public Operator<Context> {
 public:
    UpdateOpBase(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          lr_mult(OperatorBase::Arg<float>("lr_mult", 1.f)),
          decay_mult(OperatorBase::Arg<float>("decay_mult", 1.f)),
          slot(OperatorBase::Arg<string>("slot", "")) {
        CHECK(!slot.empty()) << "\nRequired a non-empty slot";
    }
    USE_OPERATOR_FUNCTIONS;

    string Slot() { return slot + "/" + Output(0)->name(); }

    float Param(const string& name) const;

    template <typename T>
    void ProcessGradients(Tensor* dX, Tensor* X);

    virtual void ComputeUpdates(Tensor* dX) = 0;

    template <typename T>
    void ApplyUpdates(Tensor* dX, Tensor* X);

    void RunOnDevice() override;

 protected:
    float lr_mult, decay_mult;
    float l2_decay, clip_thresh, scale_factor;
    string slot;
};

#define USE_UPDATER_FUNCTIONS(context) \
    using UpdateOpBase<context>::Param; \
    using UpdateOpBase<context>::Slot

}  // namespace dragon

#endif  // DRAGON_OPERATORS_UPDATE_UPDATE_OP_BASE_H_