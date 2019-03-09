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

#ifndef DRAGON_OPERATORS_VISION_NN_RESIZE_OP_H_
#define DRAGON_OPERATORS_VISION_NN_RESIZE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class NNResizeOp final : public Operator<Context> {
 public:
    NNResizeOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          fy(OperatorBase::Arg<float>("fy", -1.f)),
          fx(OperatorBase::Arg<float>("fx", -1.f)),
          shape_like_desc(OperatorBase::Arg<string>(
              "shape_like", "")),
          data_format(OperatorBase::Arg<string>(
              "data_format", "NCHW")) {
        if (data_format == "NCHW") spatial_axis = 2;
        else if (data_format == "NHWC") spatial_axis = 1;
        else LOG(FATAL) << "Unknown data format: " << data_format;
        GET_ARGUMENTS_WITH_DESC(int64_t, dsize);
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float fy, fx;
    string data_format, shape_like_desc;
    int64_t n, c, h, w, out_h, out_w, spatial_axis;
    DECLARE_ARGUMENTS_WITH_DESC(int64_t, dsize);
};

template <class Context>
class NNResizeGradientOp final : public Operator<Context> {
 public:
    NNResizeGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          data_format(OperatorBase::Arg<string>(
              "data_format", "NCHW")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    void RunWithFloat16();
    template <typename T> void RunWithType();

 protected:
    string data_format;
    int64_t n, c, h, w, out_h, out_w;
};

DEFINE_ARGUMENTS_WITH_DESC(int64_t, NNResizeOp, dsize);

}  // namespace dragon

#endif  // DRAGON_OPERATORS_VISION_NN_RESIZE_OP_H_