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

#ifndef DRAGON_OPERATORS_VISION_NN_RESIZE_OP_H_
#define DRAGON_OPERATORS_VISION_NN_RESIZE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class NNResizeOp : public Operator<Context> {
 public:
    NNResizeOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          fy(OperatorBase::GetSingleArg<float>("fy", -1.0)),
          fx(OperatorBase::GetSingleArg<float>("fx", -1.0)),
          shape_like_desc(OperatorBase::GetSingleArg<string>("shape_like", "")),
          data_format(OperatorBase::GetSingleArg<string>("data_format", "NCHW")) {
        GET_ARGUMENTS_WITH_DESC(int, dsize);
        if (data_format == "NCHW") spatial_axis = 2;
        else if (data_format == "NHWC") spatial_axis = 1;
        else LOG(FATAL) << "Unknown data format: " << data_format;
    }
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    DECLARE_ARGUMENTS_WITH_DESC(int, dsize);
    float fy, fx;
    string data_format, shape_like_desc;
    TIndex n, c, h, w, out_h, out_w, spatial_axis;
};

template <class Context>
class NNResizeGradientOp : public Operator<Context> {
 public:
    NNResizeGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          data_format(OperatorBase::GetSingleArg<string>("data_format", "NCHW")) {}
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    string data_format;
    TIndex n, c, h, w, out_h, out_w;
};

DEFINE_ARGUMENTS_WITH_DESC(int, NNResizeOp, dsize);

}    // namespace dragon

#endif    // DRAGON_OPERATORS_VISION_NN_RESIZE_OP_H_