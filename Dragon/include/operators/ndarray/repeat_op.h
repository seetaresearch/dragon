// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_NDARRAY_REPEAT_OP_H_
#define DRAGON_OPERATORS_NDARRAY_REPEAT_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class RepeatOp : public Operator<Context> {
 public:
    RepeatOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)) {
        GET_ARGUMENT_WITH_DESC(int, repeats, 1);
    }

    void RunOnDevice() override;
    template<typename T> void RunWithType();

 protected:
    DECLARE_ARGUMENT_WITH_DESC(int, repeats);
    TIndex axis, outer_dim, dim, inner_dim;
};

template <class Context>
class RepeatGradientOp : public Operator<Context> {
 public:
    RepeatGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)) {
        GET_ARGUMENT_WITH_DESC(int, repeats, 1);
    }

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
