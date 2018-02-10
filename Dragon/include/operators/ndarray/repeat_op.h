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
          axis(OperatorBase::GetSingleArg<int>("axis", -1)),
          repeats_desc(OperatorBase::GetSingleArg<string>("repeats", "")) {}

    void RunOnDevice() override;
    template<typename T> void RunWithType();

 protected:
    TIndex axis, outer_dim, dim, inner_dim, reps;
    string repeats_desc;
};

template <class Context>
class RepeatGradientOp : public Operator<Context> {
 public:
    RepeatGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)),
          repeats_desc(OperatorBase::GetSingleArg<string>("repeats", "")) {}

    void RunOnDevice() override;
    template<typename T> void RunWithType();

protected:
    TIndex axis, outer_dim, dim, inner_dim, reps;
    string repeats_desc;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_REPEAT_OP_H_
