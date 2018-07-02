// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_OPERATORS_VISION_BIAS_ADD_OP_H_
#define DRAGON_OPERATORS_VISION_BIAS_ADD_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class BiasAddOp final : public Operator<Context> {
 public:
    BiasAddOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          data_format(OperatorBase::Arg<string>("data_format", "NCHW")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex outer_dim, dim, inner_dim;
    string data_format;
};

template <class Context>
class BiasAddGradientOp final : public Operator<Context> {
 public:
    BiasAddGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          data_format(OperatorBase::Arg<string>("data_format", "NCHW")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    int outer_dim, dim, inner_dim;
    string data_format;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_VISION_BIAS_ADD_OP_H_