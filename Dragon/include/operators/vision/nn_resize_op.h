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
          fy_(OpArg<float>("fy", -1.f)),
          fx_(OpArg<float>("fx", -1.f)),
          shape_desc_(OpArg<string>(
              "shape_like", "")) {
        if (data_format() == "NCHW") axis_ = 2;
        else if (data_format() == "NHWC") axis_ = 1;
        else LOG(FATAL) << "Unknown DataFormat: " << data_format();
        GET_ARGS_WITH_DESC(int64_t, dsize);
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    float fy_, fx_;
    string shape_desc_;
    int64_t n_, c_, h_, w_;
    int64_t out_h_, out_w_, axis_;
    DECLARE_ARGS_WITH_DESC(int64_t, dsize);
};

template <class Context>
class NNResizeGradientOp final : public Operator<Context> {
 public:
    SIMPLE_CTOR_DTOR(NNResizeGradientOp);
    USE_OPERATOR_FUNCTIONS;

    void RunImplFloat16();

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    int64_t n_, c_, h_, w_, out_h_, out_w_;
};

DEFINE_ARGS_WITH_DESC(int64_t, NNResizeOp, dsize);

}  // namespace dragon

#endif  // DRAGON_OPERATORS_VISION_NN_RESIZE_OP_H_