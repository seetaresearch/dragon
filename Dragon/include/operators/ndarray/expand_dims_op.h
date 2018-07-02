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

#ifndef DRAGON_OPERATORS_NDARRAY_EXPAND_DIMS_OP_H_
#define DRAGON_OPERATORS_NDARRAY_EXPAND_DIMS_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ExpandDimsOp final : public Operator<Context> {
 public:
    ExpandDimsOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", -1)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;

 protected:
    TIndex axis;
};

template <class Context>
class ExpandDimsGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(ExpandDimsGradientOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NDARRAY_EXPAND_DIMS_OP_H_