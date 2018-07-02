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

#ifndef DRAGON_OPERATORS_NDARRAY_REPEAT_OP_H_
#define DRAGON_OPERATORS_NDARRAY_REPEAT_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class RepeatOp final : public Operator<Context> {
 public:
    RepeatOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", -1)) {
        GET_ARGUMENT_WITH_DESC(int, repeats, 1);
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template<typename T> void RunWithType();

 protected:
    DECLARE_ARGUMENT_WITH_DESC(int, repeats);
    TIndex axis, outer_dim, dim, inner_dim;
};

template <class Context>
class RepeatGradientOp final : public Operator<Context> {
 public:
    RepeatGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", -1)) {
        GET_ARGUMENT_WITH_DESC(int, repeats, 1);
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template<typename T> void RunWithType();

 protected:
    DECLARE_ARGUMENT_WITH_DESC(int, repeats);
    TIndex axis, outer_dim, dim, inner_dim, reps;
};

DEFINE_ARGUMENT_WITH_DESC(int, RepeatOp, repeats);
DEFINE_ARGUMENT_WITH_DESC(int, RepeatGradientOp, repeats);

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_REPEAT_OP_H_
