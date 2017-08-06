// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_COMMON_UTILS_OP_H_
#define DRAGON_OPERATORS_COMMON_UTILS_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class CopyOp final: public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(CopyOp);

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

template <class Context>
class AccuracyOp final: public Operator<Context> {
 public:
    AccuracyOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          top_k(OperatorBase::GetSingleArg<int>("top_k", 1)) {
        vector<int> args = OperatorBase::GetRepeatedArg<int>("ignore_labels");
        if (args.size()) {
            ignore_labels.Reshape(vector<TIndex>(1, args.size()));
            int* ignore_data = ignore_labels.mutable_data<int, CPUContext>();
            for (int i = 0; i < args.size(); i++) ignore_data[i] = args[i];
        }
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex top_k, outer_num, inner_num, classes;
    Tensor ignore_labels;
};

template <class Context>
class OneHotOp final : public Operator < Context > {
 public:
    OneHotOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          depth(OperatorBase::GetSingleArg<int>("depth", -1)),
          on_value(OperatorBase::GetSingleArg<int>("on_value", 1)),
          off_value(OperatorBase::GetSingleArg<int>("off_value", 0)) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex depth, on_value, off_value;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_COMMON_UTILS_OP_H_