// ------------------------------------------------------------
// Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// -------------------------------------------------------------

#ifndef DRAGON_OPERATORS_MISC_ACCURACY_OP_H_
#define DRAGON_OPERATORS_MISC_ACCURACY_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class AccuracyOp final: public Operator<Context> {
 public:
    AccuracyOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          top_k(OperatorBase::GetSingleArg<int>("top_k", 1)),
          axis(OperatorBase::GetSingleArg<int>("axis", 1)) {
        vector<int> args = OperatorBase::GetRepeatedArg<int>("ignore_labels");
        if (args.size()) {
            ignore_labels.Reshape(vector<TIndex>(1, args.size()));
            int* ignore_data = ignore_labels.mutable_data<int, CPUContext>();
            for (int i = 0; i < args.size(); i++) ignore_data[i] = args[i];
        }
    }
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex top_k, axis, outer_dim, inner_dim, num_classes;
    Tensor ignore_labels;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_MISC_ACCURACY_OP_H_